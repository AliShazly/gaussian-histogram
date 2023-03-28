#![feature(path_file_prefix)]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use image::{DynamicImage, GenericImageView};
use lazy_static::lazy_static;
use puruspe::{erf, inverf};
use rayon::prelude::*;
use rfd::FileDialog;
use tiff::encoder::compression::Packbits;
use tiff::encoder::{colortype, TiffEncoder, TiffValue};

const IMG_SUFFIX: &str = "gaussian";
const LUT_SUFFIX: &str = "lut";
const GAUSSIAN_AVERAGE: f64 = 0.5;

lazy_static! {
    static ref GAUSSIAN_STD: f64 = (1.0_f64 / 36.0_f64).sqrt();
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq, Copy)]
enum Output {
    Img,
    Lut,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    in_file: Option<PathBuf>,

    #[arg(short, long, default_value = "./")]
    out_dir: PathBuf,

    #[arg(long)]
    img_prefix: Option<String>,

    #[arg(long)]
    lut_prefix: Option<String>,
}

fn main() {
    if let Err(e) = _main(Args::parse()) {
        let exe_name = std::env::args()
            .next()
            .unwrap_or_else(|| "precompute".to_string());
        eprintln!("[{}] Precompute Error: {}", exe_name, e);
        std::process::exit(1);
    }
}

fn _main(args: Args) -> Result<(), anyhow::Error> {
    let cwd = std::env::current_dir().unwrap_or(PathBuf::from("."));
    let input_path = args.in_file.map(Ok).unwrap_or_else(|| {
        FileDialog::new()
            .set_title("Select input texture")
            .set_directory(cwd)
            .pick_file()
            .ok_or(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No input file specified",
            ))
    })?;
    let out_dir = path_directory(args.out_dir);
    let input_file_prefix = input_path
        .file_prefix()
        .unwrap_or(OsStr::new("Texture"))
        .to_string_lossy();
    let img_file_name = format!(
        "{}.tif",
        args.img_prefix
            .unwrap_or(format!("{}-{}", input_file_prefix, IMG_SUFFIX))
    );
    let lut_file_name = format!(
        "{}.tif",
        args.lut_prefix
            .unwrap_or(format!("{}-{}", input_file_prefix, LUT_SUFFIX))
    );

    let input_img = image::open(&input_path)?;

    println!("Processing {:?}...", input_path);
    let start = Instant::now();

    let (t, t_inv) = transform_histogram(&input_img);

    println!("Finished processing. Took {:?}", start.elapsed());

    println!(
        "Writing output to {} and {} in directory {:?}",
        img_file_name, lut_file_name, out_dir
    );

    let (width, height) = (input_img.width(), input_img.height());
    let (res1, res2) = rayon::join(
        || write_rgb::<colortype::RGB32Float>(&out_dir.join(img_file_name), width, height, &t),
        || write_rgb::<colortype::RGB8>(&out_dir.join(lut_file_name), width, 1, &t_inv),
    );
    res1.and(res2)
}

struct ChannelPixel {
    subpx_idx: usize,
    sort_idx: usize,
}

fn transform_histogram(input: &DynamicImage) -> (Vec<f32>, Vec<u8>) {
    let mut t_r = vec![0.0; (input.width() * input.height()) as usize];
    let mut t_g = vec![0.0; (input.width() * input.height()) as usize];
    let mut t_b = vec![0.0; (input.width() * input.height()) as usize];

    let mut t_inv_r = vec![0; input.width() as usize];
    let mut t_inv_g = vec![0; input.width() as usize];
    let mut t_inv_b = vec![0; input.width() as usize];

    let mut input_r = vec![0; (input.width() * input.height()) as usize];
    let mut input_g = vec![0; (input.width() * input.height()) as usize];
    let mut input_b = vec![0; (input.width() * input.height()) as usize];

    input.pixels().enumerate().for_each(|(i, (_, _, px))| {
        input_r[i] = px[0];
        input_g[i] = px[1];
        input_b[i] = px[2];
    });

    [
        (input_r, &mut t_r, &mut t_inv_r),
        (input_g, &mut t_g, &mut t_inv_g),
        (input_b, &mut t_b, &mut t_inv_b),
    ]
    .par_iter_mut()
    .for_each(|(inp, t, t_inv)| {
        let mut input_sorted: Vec<_> = inp.clone().into_iter().enumerate().collect();
        input_sorted.par_sort_unstable_by_key(|(_, val)| *val);

        let mut input_orig_order: Vec<_> = input_sorted
            .clone()
            .into_par_iter()
            .enumerate()
            .map(|(sort_idx, (subpx_idx, _))| ChannelPixel {
                subpx_idx,
                sort_idx,
            })
            .collect();
        input_orig_order.par_sort_unstable_by_key(|pixel| pixel.subpx_idx);

        t.par_iter_mut().enumerate().for_each(|(i, subpx)| {
            let sort_idx = input_orig_order[i].sort_idx;
            let u = (sort_idx as f64 + 0.5) / (input_sorted.len() as f64);
            let g = inv_cdf(u, GAUSSIAN_AVERAGE, *GAUSSIAN_STD);
            *subpx = g as f32;
        });

        t_inv.par_iter_mut().enumerate().for_each(|(i, subpx)| {
            let g = (i as f64 + 0.5) / (input.width() as f64);
            let u = cdf(g, GAUSSIAN_AVERAGE, *GAUSSIAN_STD);
            let index = (u * input_sorted.len() as f64).floor() as usize;
            let val = input_sorted[index].1;
            *subpx = val;
        });
    });

    let (out_img, lut) = rayon::join(
        || {
            let mut out_img = vec![0.0; (input.width() * input.height() * 3) as usize];
            out_img
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i, subpx)| {
                    subpx[0] = t_r[i];
                    subpx[1] = t_g[i];
                    subpx[2] = t_b[i];
                });
            out_img
        },
        || {
            let mut lut = vec![0; (input.width() * 3) as usize];
            lut.par_chunks_mut(3).enumerate().for_each(|(i, subpx)| {
                subpx[0] = t_inv_r[i];
                subpx[1] = t_inv_g[i];
                subpx[2] = t_inv_b[i];
            });
            lut
        },
    );

    (out_img, lut)
}

fn write_rgb<T>(
    path: &Path,
    width: u32,
    height: u32,
    data: &[T::Inner],
) -> Result<(), anyhow::Error>
where
    T: colortype::ColorType,
    [T::Inner]: TiffValue,
{
    let mut file = std::fs::File::create(path)?;
    TiffEncoder::new(&mut file)?
        .write_image_with_compression::<T, _>(width, height, Packbits, data)?;

    Ok(())
}

fn path_directory(path: PathBuf) -> PathBuf {
    if path.is_dir() {
        path
    } else {
        path.parent()
            .map(|x| x.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(""))
    }
}

fn cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    0.5 * (1.0 + erf((x - mu) / (sigma * 2.0_f64.sqrt())))
}

fn inv_cdf(u: f64, mu: f64, sigma: f64) -> f64 {
    sigma * 2.0_f64.sqrt() * inverf(2.0 * u - 1.0) + mu
}
