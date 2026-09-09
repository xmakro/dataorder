#![allow(dead_code, unexpected_cfgs)]
#[path = "../float-before/interleave/mod.rs"] mod before;
#[path = "../float-final/interleave/mod.rs"] mod after;
#[path = "../src/sum.rs"] mod sum;
#[path = "../sum-before.rs"] mod sum_before;
pub use dataorder::SamplingDetail;
#[derive(Default)] pub struct SamplingDiagnostics { demand:f64, start:f64, end:f64, used_tolerance:bool, clamped_uniform:bool }
use serde_json::{json,Value};
use std::hint::black_box;
use std::time::{Instant,Duration};
fn float_bits(x:f64)->u64 { if x==0.0 {0} else {x.to_bits()} }
fn convert(s:before::Sampling)->after::Sampling {
    match s {
        before::Sampling::Uniform => after::Sampling::Uniform,
        before::Sampling::DelayedLinear{start,full} => after::Sampling::ramp(start,full),
        before::Sampling::Trapezoid{start,full,fade,off} => after::Sampling::trapezoid(start,full,fade,off),
    }
}
fn configuration(k: usize, style: &str) -> (Vec<u64>, Vec<before::Sampling>) {
    let mut x = 0x2545f4914f6cdd1du64;
    let lens = (0..k)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            500_000 + x % 1_000_000
        })
        .collect();
    let sampling = (0..k)
        .map(|i| match (style, i % 10) {
            ("uniform", _) => before::Sampling::Uniform,
            ("ramps-dyadic", 3) => before::Sampling::ramp(0.125, 0.375),
            ("ramps-dyadic", 7) => before::Sampling::trapezoid(0.0, 0.25, 0.75, 1.0),
            ("ramps-binary", 3) => before::Sampling::ramp(0.1, 0.3),
            ("ramps-binary", 7) => before::Sampling::trapezoid(0.0, 0.2, 0.7, 1.0),
            ("distinct-dyadic", 3 | 7) => before::Sampling::delayed(i as f64 / (2 * k.next_power_of_two()) as f64),
            ("distinct-binary", 3 | 7) => before::Sampling::delayed(0.1 + 0.6 * i as f64 / k as f64),
            _ => before::Sampling::Uniform,
        })
        .collect();
    (lens, sampling)
}

fn paired(mut measure:impl FnMut(bool,usize)->f64,units:usize)->Value {
    let mut reps=[1usize;2];
    for (side,r) in reps.iter_mut().enumerate() {
        while measure(side==1,*r)*((*r*units) as f64) <20_000_000.0 {*r*=2;}
    }
    let mut samples=[Vec::new(),Vec::new()];
    for round in 0..5 {
        for side in if round%2==0 {[0,1]} else {[1,0]} {
            samples[side].push(measure(side==1,reps[side]));
        }
    }
    let medians:Vec<_>=samples.iter().map(|v|{let mut v=v.clone();v.sort_by(f64::total_cmp);v[2]}).collect();
    json!({"before_ns":medians[0],"after_ns":medians[1],"ratio":medians[1]/medians[0],"samples":samples,"reps":reps,"units":units})
}
fn benchmark()->Vec<Value> {
    let mut rows=Vec::new();
    for style in ["uniform","ramps-dyadic","ramps-binary","distinct-binary"] {
        let (lens,bs)=configuration(1000,style);
        let cs:Vec<_>=bs.iter().copied().map(convert).collect();
        let b=before::Interleave::with_sampling(&lens,&bs).unwrap();
        let a=after::Interleave::with_sampling(&lens,&cs).unwrap();
        let n=b.len();
        let build=paired(|side,reps|{
            let t=Instant::now();
            for _ in 0..reps {
                if side { black_box(after::Interleave::with_sampling(black_box(&lens),black_box(&cs)).unwrap()); }
                else { black_box(before::Interleave::with_sampling(black_box(&lens),black_box(&bs)).unwrap()); }
            }
            t.elapsed().as_secs_f64()*1e9/reps as f64
        },1);
        let seek=paired(|side,reps|{
            let t=Instant::now();
            for r in 0..reps {
                let p=black_box((r as u64*7_919_337_017)%n);
                if side {black_box(a.iter(p..n).next());} else {black_box(b.iter(p..n).next());}
            }
            t.elapsed().as_secs_f64()*1e9/reps as f64
        },1);
        for (num,den) in [(1,5),(1,2),(4,5)] {
            let p=n*num/den;
            let (mut bi,mut ai)=(b.iter(p..n),a.iter(p..n));
            let walk=paired(|side,reps|{
                let mut elapsed=Duration::ZERO;
                for _ in 0..reps {
                    if side {ai.seek(p..n);} else {bi.seek(p..n);}
                    let t=Instant::now();
                    let mut check=0u64;
                    if side {for _ in 0..1024 {let (s,j)=ai.next().unwrap();check=check.wrapping_add(s as u64^j);}}
                    else {for _ in 0..1024 {let (s,j)=bi.next().unwrap();check=check.wrapping_add(s as u64^j);}}
                    elapsed+=t.elapsed();black_box(check);
                }
                elapsed.as_secs_f64()*1e9/(reps*1024) as f64
            },1024);
            println!("{style}@{num}/{den}: {walk}");
            rows.push(json!({"style":style,"progress":[num,den],"walk":walk,"seek":seek,"build":build}));
        }
    }
    rows
}
fn compatibility()->Value {
    let mut cases=0usize;
    let mut differing=0usize;
    let mut first=Value::Null;
    for scheduled in (1..400).step_by(2) {
        for uniform in [scheduled+1,scheduled*2,scheduled*3,scheduled*10,1<<30] {
            for off in [0.5,0.75,1.0] {
                let lens=[uniform,scheduled];
                let bs=[before::Sampling::Uniform,before::Sampling::trapezoid(0.0,0.0,0.0,off)];
                let cs=bs.map(convert);
                let Ok(b)=before::Interleave::with_sampling(&lens,&bs) else {continue;};
                let a=after::Interleave::with_sampling(&lens,&cs).unwrap();
                let n=b.len();
                cases+=1;
                for j in 0..scheduled {
                    let q=(j as f64+0.75)/scheduled as f64;
                    let p=(off*(1.0-(1.0-q).sqrt())*n as f64) as u64;
                    let start=p.saturating_sub(5);
                    let end=(p+6).min(n);
                    let old:Vec<_>=b.iter(start..end).collect();
                    let new:Vec<_>=a.iter(start..end).collect();
                    if old!=new {
                        differing+=1;
                        if first.is_null() {first=json!({"lens":lens,"off":off,"start":start,"before":old,"after":new});}
                        break;
                    }
                }
            }
        }
    }
    json!({"cases":cases,"differing_cases":differing,"first_difference":first})
}

fn main() {
    let mut rows=Vec::new();
    for k in [3,100,1000,10000] {
        for style in ["uniform","ramps-binary","distinct-binary"] {
            let (lens,bs)=if k==3 && style!="uniform" {
                (vec![2_000_000,100_000,100_000], if style=="ramps-binary" {
                    vec![before::Sampling::Uniform,before::Sampling::ramp(0.1,0.3),before::Sampling::trapezoid(0.0,0.2,0.7,1.0)]
                } else {vec![before::Sampling::Uniform,before::Sampling::delayed(0.1),before::Sampling::delayed(0.6)]})
            } else {configuration(k,style)};
            let cs:Vec<_>=bs.iter().copied().map(convert).collect();
            let build=paired(|side,reps|{
                let t=Instant::now();
                for _ in 0..reps {
                    if side {black_box(after::Interleave::with_sampling(black_box(&lens),black_box(&cs)).unwrap());}
                    else {black_box(before::Interleave::with_sampling(black_box(&lens),black_box(&bs)).unwrap());}
                }
                t.elapsed().as_secs_f64()*1e9/reps as f64
            },1);
            println!("{k} {style} {build}");
            rows.push(json!({"k":k,"style":style,"build":build}));
        }
    }
    std::fs::write("target/final-build-results.json",serde_json::to_string_pretty(&rows).unwrap()).unwrap();
}
