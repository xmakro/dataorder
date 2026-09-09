
#[cfg(test)]
mod inverse_audit {
    use super::Profile;
    #[test]
    fn rising_inverse_extremes() {
        let mut rows=Vec::new();
        let mut checked=0usize;
        for (start,end) in [(0.0,1.0),(0.125,0.75),(0.0,1e-300),(0.5,0.5f64.next_up())] {
            for ratio in [0.0,1e-150,1e-30,1e-12,0.01,0.5,0.99,1.0f64.next_down()] {
                let (r0,r1)=(ratio,1.0);
                let p=Profile::from_rates([(start,end,r0,r1)]);
                if !p.is_finite() {continue;}
                assert_eq!(p.quantile(0.0,&mut 0),start);
                let mass=(r0+r1)/2.0*(end-start);
                for q in [0.0,1e-25,1e-15,1e-9,0.001,0.1,0.5,0.9,0.999,1.0f64.next_down(),1.0] {
                    let mut y=mass*q;
                    let mut previous=p.quantile(y,&mut 0);
                    for i in 0..64 {
                        let actual=p.quantile(y,&mut 0);
                        assert!(actual>=previous && actual.is_finite(),"nonmonotone {start} {end} {r0} {y}");
                        if i==0 || i==63 {rows.push(serde_json::json!([start,end,r0,r1,y,actual]));}
                        previous=actual;
                        y=y.next_up().min(mass);
                        checked+=1;
                    }
                }
            }
        }
        std::fs::write("target/float-numeric.json",serde_json::to_string(&rows).unwrap()).unwrap();
        println!("checked {checked} adjacent quantiles; exported {} independent-reference inputs",rows.len());
    }
}
