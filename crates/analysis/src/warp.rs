//! Warping a raster onto a north-up grid through a control-point fit ([`crate::georef::Fit`]).
//!
//! `GeoTransform` has no rotation terms, so an image that a fit turns, shears or bends can only be
//! placed by resampling it. Every cell of the output grid is taken back through the fit (after
//! reprojection, when the output's coordinate system isn't the one the control points are in) to a
//! position in the image and sampled there; cells whose position falls outside the image are
//! no data.
//!
//! The way back to the image is computed exactly at every [`STEP`]th cell of a row and linearly
//! between, as GDAL's approximate transformer does: where the midpoint of a stretch is off by more
//! than [`TOLERANCE`] image pixels, the whole stretch is computed exactly.

use rustgis_data::raster::{ColorHint, Crs, GeoTransform, MAX_SAMPLES, Raster, Transformer};
use rustgis_geom::Coord;

use crate::georef::Fit;

/// Cells between exact computations of the way back to the image.
const STEP: usize = 8;
/// The most, in image pixels, a linearly placed midpoint may be off.
const TOLERANCE: f64 = 0.125;
/// Points along each side of the image when its footprint is measured.
const EDGE_POINTS: usize = 64;
/// Divisions of the image each way for the lattice of points inside it.
const INTERIOR_POINTS: usize = 32;
/// What one output column costs in working buffers, in `f32` samples: its image position (24
/// bytes) and its share of the exact nodes (4 bytes).
const COLUMN_COST: f64 = 7.0;

/// How cells are sampled from the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resampling {
    /// The pixel the position falls in: values are kept, which is right for classes.
    Nearest,
    /// The four nearest pixel centres, by distance.
    Bilinear,
    /// Catmull-Rom over the sixteen nearest, kept within the band's range; where one of them has
    /// no value, bilinear.
    Cubic,
}

impl Resampling {
    pub fn parse(s: &str) -> Option<Resampling> {
        match s.trim().to_ascii_lowercase().as_str() {
            "nearest" => Some(Resampling::Nearest),
            "bilinear" => Some(Resampling::Bilinear),
            "cubic" => Some(Resampling::Cubic),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Resampling::Nearest => "nearest",
            Resampling::Bilinear => "bilinear",
            Resampling::Cubic => "cubic",
        }
    }
}

/// What to warp onto.
#[derive(Clone, Debug)]
pub struct Options {
    /// The output grid's coordinate system.
    pub crs: Crs,
    /// The coordinate system of the fit's map coordinates.
    pub points_crs: Crs,
    pub resampling: Resampling,
    /// Cell size in the output's units; about the image pixel's size on the ground when `None`.
    pub cell: Option<f64>,
    pub name: String,
}

/// Map coordinates of the fit, to and from the output's coordinate system.
struct Reproject {
    from: Transformer,
    to: Transformer,
    same: bool,
}

impl Reproject {
    fn new(points: Crs, out: Crs) -> Result<Reproject, String> {
        Ok(Reproject { from: points.transformer()?, to: out.transformer()?, same: points == out })
    }

    /// Fit coordinates to output coordinates.
    fn forward(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if self.same {
            return (x.is_finite() && y.is_finite()).then_some((x, y));
        }
        let c = self.to.from_lonlat(self.from.to_lonlat(Coord::new(x, y))?)?;
        Some((c.x, c.y))
    }

    /// Output coordinates to fit coordinates.
    fn back(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if self.same {
            return (x.is_finite() && y.is_finite()).then_some((x, y));
        }
        let c = self.from.from_lonlat(self.to.to_lonlat(Coord::new(x, y))?)?;
        Some((c.x, c.y))
    }
}

/// Reads pixels at fractional image positions.
struct Sampler<'a> {
    r: &'a Raster,
    method: Resampling,
    /// The band whose value weighs the others (alpha), if the image has one.
    alpha: Option<usize>,
    /// Whole-number pixel types are rounded.
    round: bool,
}

impl Sampler<'_> {
    /// The value of band `b` at index `idx`, if it is a real measurement.
    fn at(&self, b: usize, idx: usize) -> Option<f32> {
        let v = *self.r.bands.get(b)?.data.get(idx)?;
        self.r.is_valid(v).then_some(v)
    }

    /// Every band at image position (`c`, `row`), in pixels from the top left corner; NaN where
    /// there's no value. The position must be inside the image.
    fn sample(&self, c: f64, row: f64, out: &mut [f32]) {
        let (w, h) = (i64::from(self.r.width), i64::from(self.r.height));
        let index = |i: i64, j: i64| (j.clamp(0, h - 1) * w + i.clamp(0, w - 1)) as usize;
        if self.method == Resampling::Nearest {
            let idx = index(c.floor() as i64, row.floor() as i64);
            for (b, o) in out.iter_mut().enumerate() {
                *o = self.at(b, idx).unwrap_or(f32::NAN);
            }
            return;
        }
        // Pixel centres are at half integers.
        let (u, v) = (c - 0.5, row - 0.5);
        let (i0, j0) = (u.floor() as i64, v.floor() as i64);
        let (fu, fv) = (u - i0 as f64, v - j0 as f64);
        for (b, o) in out.iter_mut().enumerate() {
            let cubic = (self.method == Resampling::Cubic).then(|| self.cubic(b, &index, i0, j0, fu, fv)).flatten();
            *o = cubic.unwrap_or_else(|| self.bilinear(b, &index, i0, j0, fu, fv));
            if self.round && o.is_finite() {
                *o = o.round();
            }
        }
    }

    fn bilinear(&self, b: usize, index: &dyn Fn(i64, i64) -> usize, i0: i64, j0: i64, fu: f64, fv: f64) -> f32 {
        let weighted = self.alpha.filter(|a| *a != b);
        let (mut sum, mut weight) = (0.0f64, 0.0f64);
        for (di, dj, w) in [(0, 0, (1.0 - fu) * (1.0 - fv)), (1, 0, fu * (1.0 - fv)), (0, 1, (1.0 - fu) * fv), (1, 1, fu * fv)] {
            let idx = index(i0 + di, j0 + dj);
            let Some(v) = self.at(b, idx) else { continue };
            // Colours count as much as their pixel is opaque.
            let k = weighted.map_or(1.0, |a| self.at(a, idx).map_or(0.0, f64::from));
            sum += f64::from(v) * w * k;
            weight += w * k;
        }
        if weight > 1e-12 { (sum / weight) as f32 } else { f32::NAN }
    }

    /// Catmull-Rom over the sixteen pixels around the position; `None` where one has no value,
    /// and for a colour band unless all sixteen have the same alpha above 0 (bilinear, which
    /// weighs colours by alpha, takes the rest).
    fn cubic(&self, b: usize, index: &dyn Fn(i64, i64) -> usize, i0: i64, j0: i64, fu: f64, fv: f64) -> Option<f32> {
        let kernel = |t: f64| {
            [
                (-t * t * t + 2.0 * t * t - t) / 2.0,
                (3.0 * t * t * t - 5.0 * t * t + 2.0) / 2.0,
                (-3.0 * t * t * t + 4.0 * t * t + t) / 2.0,
                (t * t * t - t * t) / 2.0,
            ]
        };
        let (wx, wy) = (kernel(fu), kernel(fv));
        let weighted = self.alpha.filter(|a| *a != b);
        let mut alpha = None;
        let mut sum = 0.0f64;
        for (dj, wj) in wy.iter().enumerate() {
            for (di, wi) in wx.iter().enumerate() {
                let idx = index(i0 + di as i64 - 1, j0 + dj as i64 - 1);
                if let Some(a) = weighted {
                    // A see-through pixel's colour mustn't count.
                    let k = self.at(a, idx)?;
                    if k <= 0.0 || *alpha.get_or_insert(k) != k {
                        return None;
                    }
                }
                let v = self.at(b, idx)?;
                sum += f64::from(v) * wi * wj;
            }
        }
        // Kept within the band's range (a range that is missing or reversed leaves it be).
        let s = &self.r.bands.get(b)?.stats;
        Some(if s.min <= s.max { sum.clamp(s.min, s.max) } else { sum } as f32)
    }
}

/// Warps `r` onto a north-up grid in `o.crs` through `fit`, whose pixel positions are those of
/// `r` (in the full image's pixels, when `r` is an overview of it) and whose map coordinates are
/// in `o.points_crs`. The grid covers the warped image's footprint. A colour or grey byte raster
/// with neither alpha nor no-data gets an alpha band (0–255), so the cells outside the image are
/// see-through. Geographic output cells are square in degrees.
pub fn warp(r: &Raster, fit: &Fit, o: &Options) -> Result<Raster, String> {
    let re = Reproject::new(o.points_crs, o.crs)?;
    let (w, h) = (f64::from(r.width), f64::from(r.height));
    // An overview's pixels are bigger than the full image's, which the fit is in.
    let (sx, sy) = r.overview.map_or((1.0, 1.0), |(fw, fh)| (f64::from(fw) / w, f64::from(fh) / h));
    let place = |c: f64, row: f64| fit.to_map(c * sx, row * sy).and_then(|(x, y)| re.forward(x, y));

    // The footprint: the image's edge and a lattice inside it, placed (a pole inside the image
    // is no extreme of its edge).
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    let mut placed = 0;
    let mut add = |c: f64, row: f64| {
        if let Some((x, y)) = place(c, row) {
            (min_x, min_y, max_x, max_y) = (min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y));
            placed += 1;
        }
    };
    for i in 0..=EDGE_POINTS {
        let t = i as f64 / EDGE_POINTS as f64;
        for (c, row) in [(t * w, 0.0), (t * w, h), (0.0, t * h), (w, t * h)] {
            add(c, row);
        }
    }
    for i in 0..=INTERIOR_POINTS {
        for j in 0..=INTERIOR_POINTS {
            add(i as f64 / INTERIOR_POINTS as f64 * w, j as f64 / INTERIOR_POINTS as f64 * h);
        }
    }
    if placed < 4 || ![min_x, min_y, max_x, max_y].iter().all(|v| v.is_finite()) {
        return Err("the image's edge can't be placed in the output coordinate system".into());
    }
    if o.crs.is_geographic() && max_x - min_x > 180.0 {
        return Err("the warped image spans more than 180° of longitude (it crosses the antimeridian, or the fit sends part of it far away): give a projected crs".into());
    }
    if max_x <= min_x || max_y <= min_y {
        return Err("the fit puts the image on a line, with no area".into());
    }

    let nb = r.bands.len();
    let alpha_src = match r.color {
        ColorHint::Rgb { alpha: true } if nb >= 4 => Some(3),
        ColorHint::GrayAlpha if nb >= 2 => Some(1),
        _ => None,
    };
    let rgb = matches!(r.color, ColorHint::Rgb { .. }) && nb >= 3;
    let gray = matches!(r.color, ColorHint::Gray) && nb == 1 && r.data_type == "UInt8";
    let add_alpha = (rgb || gray) && alpha_src.is_none() && r.nodata.is_none();
    let out_bands = nb + usize::from(add_alpha);

    // The cell: asked for, or the image pixel's size on the ground (the root of its area, where
    // the fit turns or skews it), coarser if the grid would hold too many samples.
    let cells = |cell: f64| -> (f64, f64) { (((max_x - min_x) / cell).ceil().max(1.0), ((max_y - min_y) / cell).ceil().max(1.0)) };
    // The samples held, and the buffers a row needs (they add up on a long thin grid).
    let load = |(ow, oh): (f64, f64)| ow * (oh * out_bands as f64 + COLUMN_COST);
    let too_many = |n: (f64, f64)| load(n) > MAX_SAMPLES as f64;
    let cell = match o.cell {
        Some(c) if !(c.is_finite() && c > 0.0) => return Err("cellSize must be a positive number".into()),
        Some(c) => {
            if too_many(cells(c)) {
                return Err(format!("a cell size of {c} makes a grid of more than {MAX_SAMPLES} samples: give a larger cellSize"));
            }
            c
        }
        None => {
            let (cx, cy) = (w / 2.0, h / 2.0);
            let (a, b, d) = (place(cx, cy), place(cx + 1.0, cy), place(cx, cy + 1.0));
            let (Some(a), Some(b), Some(d)) = (a, b, d) else {
                return Err("the source pixel's size can't be worked out: give a cellSize".into());
            };
            let mut cell = ((b.0 - a.0) * (d.1 - a.1) - (d.0 - a.0) * (b.1 - a.1)).abs().sqrt();
            if !(cell.is_finite() && cell > 0.0) {
                return Err("the source pixel's size can't be worked out: give a cellSize".into());
            }
            // Each step at least 1% coarser, so a grid rounded up past the limit still shrinks.
            for _ in 0..64 {
                let n = cells(cell);
                if !too_many(n) {
                    break;
                }
                cell *= (load(n) / MAX_SAMPLES as f64).sqrt().max(1.01);
            }
            if too_many(cells(cell)) {
                return Err("the warped image is too large to hold: give a cellSize".into());
            }
            cell
        }
    };
    let (ow, oh) = cells(cell);
    let (ow, oh) = (ow as usize, oh as usize);

    let method = if matches!(r.color, ColorHint::Palette(_)) { Resampling::Nearest } else { o.resampling };
    let sampler = Sampler { r, method, alpha: alpha_src, round: !r.data_type.starts_with("Float") };
    // Cells with no value: no-data where the raster has some. Colours with alpha are zero there.
    let clear = alpha_src.is_some() || add_alpha;
    let outside = if clear { 0.0 } else { r.nodata.map_or(f32::NAN, |d| d as f32) };
    // Alpha is read as 0–255 whatever the pixel type.
    let opaque = 255.0;

    let mut bands: Vec<Vec<f32>> = vec![vec![outside; ow * oh]; out_bands];
    let mut coords: Vec<Option<(f64, f64)>> = vec![None; ow];
    let mut nodes: Vec<(usize, Option<(f64, f64)>)> = Vec::new();
    let mut px = vec![f32::NAN; nb];
    // The image position of an output coordinate.
    let source = |x: f64, y: f64| -> Option<(f64, f64)> {
        let (px, py) = re.back(x, y)?;
        let (c, row) = fit.to_pixel(px, py)?;
        let (c, row) = (c / sx, row / sy);
        (c.is_finite() && row.is_finite()).then_some((c, row))
    };
    for oy in 0..oh {
        let y = max_y - (oy as f64 + 0.5) * cell;
        let x_at = |ox: usize| min_x + (ox as f64 + 0.5) * cell;
        nodes.clear();
        let mut ox = 0;
        loop {
            nodes.push((ox, source(x_at(ox), y)));
            if ox + 1 >= ow {
                break;
            }
            ox = (ox + STEP).min(ow - 1);
        }
        for pair in nodes.windows(2) {
            let [(a, pa), (b, pb)] = pair else { continue };
            let linear = match (pa, pb) {
                (Some(pa), Some(pb)) if b - a >= 2 => {
                    let mid = a + (b - a) / 2;
                    let t = (mid - a) as f64 / (b - a) as f64;
                    let guess = (pa.0 + (pb.0 - pa.0) * t, pa.1 + (pb.1 - pa.1) * t);
                    source(x_at(mid), y).is_some_and(|m| (m.0 - guess.0).abs().max((m.1 - guess.1).abs()) <= TOLERANCE)
                }
                (Some(_), Some(_)) => true,
                _ => false,
            };
            for i in *a..*b {
                let t = (i - a) as f64 / (b - a) as f64;
                let c = match (linear, pa, pb) {
                    (true, Some(pa), Some(pb)) => Some((pa.0 + (pb.0 - pa.0) * t, pa.1 + (pb.1 - pa.1) * t)),
                    _ => source(x_at(i), y),
                };
                if let Some(slot) = coords.get_mut(i) {
                    *slot = c;
                }
            }
        }
        if let Some((last, p)) = nodes.last()
            && let Some(c) = coords.get_mut(*last)
        {
            *c = *p;
        }
        for (ox, c) in coords.iter().enumerate() {
            let Some((c, row)) = *c else { continue };
            if !(c >= 0.0 && row >= 0.0 && c < w && row < h) {
                continue;
            }
            sampler.sample(c, row, &mut px);
            let at = oy * ow + ox;
            for (band, v) in bands.iter_mut().zip(&px) {
                if let Some(o) = band.get_mut(at) {
                    // A cell with no value is no-data (or see-through) like one outside.
                    *o = if v.is_nan() { if clear { 0.0 } else { outside } } else { *v };
                }
            }
            if add_alpha && let Some(o) = bands.get_mut(nb).and_then(|b| b.get_mut(at)) {
                *o = opaque;
            }
        }
    }

    let color = match (add_alpha, rgb) {
        (true, true) => ColorHint::Rgb { alpha: true },
        (true, false) => ColorHint::GrayAlpha,
        _ => r.color.clone(),
    };
    let transform = GeoTransform { origin_x: min_x, origin_y: max_y, pixel_width: cell, pixel_height: -cell };
    let mut out = Raster::new(o.name.clone(), ow as u32, oh as u32, bands, transform, o.crs, r.nodata, color, r.data_type.clone())
        .map_err(|e| e.to_string())?;
    for (i, band) in out.bands.iter_mut().enumerate() {
        band.name = r.bands.get(i).map_or_else(|| "Alpha".to_string(), |b| b.name.clone());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::georef::{self, Gcp, Kind};

    fn scan(w: u32, h: u32, bands: Vec<Vec<f32>>, color: ColorHint, data_type: &str) -> Raster {
        let gt = GeoTransform { origin_x: 0.0, origin_y: f64::from(h), pixel_width: 1.0, pixel_height: -1.0 };
        Raster::new("scan", w, h, bands, gt, Crs::LonLat, None, color, data_type).unwrap()
    }

    fn opts(cell: Option<f64>, resampling: Resampling) -> Options {
        Options { crs: Crs::from_epsg(32613).unwrap(), points_crs: Crs::from_epsg(32613).unwrap(), resampling, cell, name: "w".into() }
    }

    fn gcps(w: f64, h: f64, f: impl Fn(f64, f64) -> (f64, f64)) -> Vec<Gcp> {
        let mut v = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let (col, row) = (w * f64::from(i) / 3.0, h * f64::from(j) / 3.0);
                let (x, y) = f(col, row);
                v.push(Gcp { col, row, x, y, on: true });
            }
        }
        v
    }

    #[test]
    fn a_bent_fit_is_followed_between_exact_steps() {
        // x runs ahead of the column by 0.1 col²: bent enough that the linear steps between
        // exact computations are too far off in places, and the exact way back is used there.
        let (w, h) = (60u32, 40u32);
        let data: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 + 0.5).collect();
        let r = scan(w, h, vec![data], ColorHint::Gray, "Float32");
        let map = |c: f64, rw: f64| (500_000.0 + 10.0 * c + 0.1 * c * c, 4_000_000.0 - 10.0 * rw);
        let fit = georef::fit(Kind::Polynomial2, &gcps(60.0, 40.0, map)).unwrap();
        let out = warp(&r, &fit, &opts(Some(5.0), Resampling::Bilinear)).unwrap();
        let mut checked = 0;
        for row in (0..i64::from(out.height)).step_by(11) {
            for col in 0..i64::from(out.width) {
                let p = out.transform.to_world(col as f64 + 0.5, row as f64 + 0.5);
                // The fit's own way back, computed for this cell alone (polynomial fits are
                // fitted in each direction, so it isn't quite the inverse of the way forward).
                let Some((c, _)) = fit.to_pixel(p.x, p.y) else { continue };
                if (1.0..59.0).contains(&c) {
                    let v = out.value(0, col, row).unwrap();
                    assert!((f64::from(v) - c).abs() <= TOLERANCE + 1e-3, "cell {col},{row}: {v} against {c}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 100);
    }

    #[test]
    fn a_colour_image_gets_an_alpha_band_and_the_outside_is_see_through() {
        let (w, h) = (40u32, 40u32);
        let n = (w * h) as usize;
        let r = scan(w, h, vec![vec![200.0; n], vec![100.0; n], vec![50.0; n]], ColorHint::Rgb { alpha: false }, "UInt8");
        let (sin, cos) = 40f64.to_radians().sin_cos();
        let map = |c: f64, rw: f64| (500_000.0 + c * cos + rw * sin, 4_000_000.0 + c * sin - rw * cos);
        let fit = georef::fit(Kind::Similarity, &gcps(40.0, 40.0, map)).unwrap();
        let out = warp(&r, &fit, &opts(None, Resampling::Bilinear)).unwrap();
        assert_eq!((out.band_count(), out.color.clone()), (4, ColorHint::Rgb { alpha: true }));
        assert_eq!(out.value(3, 0, 0), Some(0.0), "the box's corner is outside the image");
        let (mid_c, mid_r) = (i64::from(out.width) / 2, i64::from(out.height) / 2);
        assert_eq!([0, 1, 2, 3].map(|b| out.value(b, mid_c, mid_r)), [Some(200.0), Some(100.0), Some(50.0), Some(255.0)]);
        // The image's cells keep whole-number values.
        assert!(out.bands.iter().all(|b| b.data.iter().all(|v| v.fract() == 0.0)));
    }

    #[test]
    fn cubic_stays_within_the_bands_range_at_an_edge() {
        let (w, h) = (30u32, 30u32);
        let data: Vec<f32> = (0..w * h).map(|i| if i % w < 15 { 0.0 } else { 100.0 }).collect();
        let r = scan(w, h, vec![data], ColorHint::Gray, "Float32");
        let map = |c: f64, rw: f64| (500_000.0 + 7.0 * c, 4_000_000.0 - 7.0 * rw);
        let fit = georef::fit(Kind::Affine, &gcps(30.0, 30.0, map)).unwrap();
        let out = warp(&r, &fit, &opts(Some(2.0), Resampling::Cubic)).unwrap();
        let (lo, hi) = out.bands[0].data.iter().filter(|v| v.is_finite()).fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(lo >= 0.0 && hi <= 100.0, "{lo}..{hi}");
        // Nearest keeps only the two values.
        let out = warp(&r, &fit, &opts(Some(2.0), Resampling::Nearest)).unwrap();
        assert!(out.bands[0].data.iter().filter(|v| v.is_finite()).all(|v| *v == 0.0 || *v == 100.0));
    }

    #[test]
    fn cubic_takes_no_colour_from_see_through_pixels() {
        // Opaque red on the left, see-through green on the right.
        let (w, h) = (30u32, 30u32);
        let half = |a: f32, b: f32| (0..w * h).map(|i| if i % w < 15 { a } else { b }).collect::<Vec<f32>>();
        let bands = vec![half(255.0, 0.0), half(0.0, 255.0), half(0.0, 0.0), half(255.0, 0.0)];
        let r = scan(w, h, bands, ColorHint::Rgb { alpha: true }, "UInt8");
        let map = |c: f64, rw: f64| (500_000.0 + 7.0 * c, 4_000_000.0 - 7.0 * rw);
        let fit = georef::fit(Kind::Affine, &gcps(30.0, 30.0, map)).unwrap();
        let out = warp(&r, &fit, &opts(Some(2.0), Resampling::Cubic)).unwrap();
        let mut edge = 0;
        for at in 0..out.bands[3].data.len() {
            let a = out.bands[3].data[at];
            if a > 0.0 {
                let rgb = [0, 1, 2].map(|b| out.bands[b].data[at]);
                assert_eq!(rgb, [255.0, 0.0, 0.0], "cell {at}, alpha {a}");
                edge += usize::from(a < 255.0);
            }
        }
        assert!(edge > 0, "some cells are partly see-through");
    }

    #[test]
    fn no_data_is_kept_outside_and_skipped_inside() {
        let gt = GeoTransform { origin_x: 0.0, origin_y: 4.0, pixel_width: 1.0, pixel_height: -1.0 };
        let mut data = vec![10.0f32; 16];
        data[5] = -9999.0;
        let r = Raster::new("nd", 4, 4, vec![data], gt, Crs::LonLat, Some(-9999.0), ColorHint::Gray, "Float32").unwrap();
        let (sin, cos) = 25f64.to_radians().sin_cos();
        let map = |c: f64, rw: f64| (500_000.0 + (c * cos + rw * sin) * 10.0, 4_000_000.0 + (c * sin - rw * cos) * 10.0);
        let fit = georef::fit(Kind::Similarity, &gcps(4.0, 4.0, map)).unwrap();
        let out = warp(&r, &fit, &opts(Some(2.0), Resampling::Bilinear)).unwrap();
        assert_eq!(out.nodata, Some(-9999.0));
        assert_eq!(out.bands[0].data[0], -9999.0, "outside the image: the raster's no-data value");
        // Bilinear skips the no-data pixel: nothing but 10 and no-data comes out.
        assert!(out.bands[0].data.iter().all(|v| *v == 10.0 || *v == -9999.0));
    }

    #[test]
    fn hostile_fits_and_options_are_refused() {
        let r = scan(4, 4, vec![vec![1.0; 16]], ColorHint::Gray, "Float32");
        let map = |c: f64, rw: f64| (500_000.0 + c, 4_000_000.0 - rw);
        let fit = georef::fit(Kind::Affine, &gcps(4.0, 4.0, map)).unwrap();
        for cell in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-12, 1e-300] {
            assert!(warp(&r, &fit, &opts(Some(cell), Resampling::Nearest)).is_err(), "{cell}");
        }
        // Enormous coordinates: no panic, whatever the answer.
        let huge = |c: f64, rw: f64| (c * 1e300, rw * 1e300);
        if let Ok(fit) = georef::fit(Kind::Affine, &gcps(4.0, 4.0, huge)) {
            let _ = warp(&r, &fit, &opts(None, Resampling::Cubic));
            let _ = warp(&r, &fit, &opts(Some(1.0), Resampling::Cubic));
        }
    }

    #[test]
    fn a_grey_byte_image_becomes_grey_with_alpha_and_every_type_is_opaque_at_255() {
        let (sin, cos) = 30f64.to_radians().sin_cos();
        let map = |c: f64, rw: f64| (500_000.0 + (c * cos + rw * sin) * 10.0, 4_000_000.0 + (c * sin - rw * cos) * 10.0);
        let fit = georef::fit(Kind::Similarity, &gcps(20.0, 20.0, map)).unwrap();
        let r = scan(20, 20, vec![vec![90.0; 400]], ColorHint::Gray, "UInt8");
        let out = warp(&r, &fit, &opts(None, Resampling::Bilinear)).unwrap();
        assert_eq!((out.band_count(), out.color.clone()), (2, ColorHint::GrayAlpha));
        let (c, rw) = (i64::from(out.width) / 2, i64::from(out.height) / 2);
        assert_eq!((out.value(0, c, rw), out.value(1, c, rw), out.value(1, 0, 0)), (Some(90.0), Some(255.0), Some(0.0)));
        // A 16-bit colour image: alpha is 255 too, as the renderer reads it.
        let n = 400;
        let r = scan(20, 20, vec![vec![60_000.0; n], vec![100.0; n], vec![7.0; n]], ColorHint::Rgb { alpha: false }, "UInt16");
        let out = warp(&r, &fit, &opts(None, Resampling::Bilinear)).unwrap();
        assert_eq!(out.value(3, c, rw), Some(255.0));
    }
}
