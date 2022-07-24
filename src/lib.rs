use std::ops::{Add, Mul, Sub};

/// invert a row-major square matrix
pub fn invert(matrix: Vec<f64>) -> Vec<f64> {
    let tmp = matrix.len() as u16;
    let mut uninverted = MatrixMath {
        vals: matrix,
        original_size: tmp,
    };
    uninverted.fill_identity();
    let mut inverted = uninverted.invert_inner();
    inverted.trim_identity();
    return inverted.vals;
}

#[derive(Default, Debug, Clone, PartialEq, PartialOrd)]
struct MatrixMath {
    vals: Vec<f64>,
    original_size: u16,
}

impl Mul for MatrixMath {
    type Output = MatrixMath;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret: Vec<f64> = vec![0.0; self.vals.len()];
        let square: usize = (self.vals.len() as f64).sqrt() as usize;
        let mut a_list: Vec<f64> = vec![0.0; square];
        let mut b_list: Vec<f64> = vec![0.0; square];
        for i in 0..square {
            for j in 0..square {
                for k in 0..square {
                    a_list[k] = self.vals[i * square + k];
                    b_list[k] = rhs.vals[k * square + j];
                }
                for k in 0..square {
                    a_list[k] *= b_list[k];
                }
                let mut acc: f64 = 0.0;
                for k in 0..square {
                    acc += a_list[k];
                }
                ret[i * square + j] = acc;
            }
        }
        MatrixMath {
            vals: ret,
            original_size: self.original_size,
        }
    }
}

impl Add for MatrixMath {
    type Output = MatrixMath;
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret: Vec<f64> = vec![0.0; self.vals.len()];
        let square: usize = (self.vals.len() as f64).sqrt() as usize;
        for i in 0..square {
            for j in 0..square {
                ret[i * square + j] = self.vals[i * square + j] + rhs.vals[i * square + j];
            }
        }
        MatrixMath {
            vals: ret,
            original_size: self.original_size,
        }
    }
}

impl Sub for MatrixMath {
    type Output = MatrixMath;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret: Vec<f64> = vec![0.0; self.vals.len()];
        let square: usize = (self.vals.len() as f64).sqrt() as usize;
        for i in 0..square {
            for j in 0..square {
                ret[i * square + j] = self.vals[i * square + j] - rhs.vals[i * square + j];
            }
        }
        MatrixMath {
            vals: ret,
            original_size: self.original_size,
        }
    }
}

fn quarters_merge(e: MatrixMath, f: MatrixMath, g: MatrixMath, h: MatrixMath) -> MatrixMath {
    let quarter_array_size: usize = e.vals.len();
    let square: usize = (quarter_array_size as f64).sqrt() as usize;
    let rowsize: usize = ((quarter_array_size * 4) as f64).sqrt() as usize;
    let half_offset: usize = rowsize * rowsize / 2;
    let mut ret: Vec<f64> = vec![0.0; quarter_array_size * 4];
    for i in 0..square {
        for j in 0..square {
            ret[i * rowsize + j] = e.vals[i * square + j];
        }
        for j in 0..square {
            ret[i * rowsize + j + square] = f.vals[i * square + j];
        }
    }
    for i in 0..square {
        for j in 0..square {
            ret[half_offset + i * rowsize + j] = g.vals[i * square + j];
        }
        for j in 0..square {
            ret[half_offset + i * rowsize + square + j] = h.vals[i * square + j];
        }
    }
    MatrixMath {
        vals: ret,
        original_size: e.original_size,
    }
}

impl MatrixMath {
    #[cfg(test)]
    fn new(vals: Vec<f64>, original_size: u16) -> MatrixMath {
        MatrixMath {
            vals,
            original_size,
        }
    }

    fn sign_swap(self) -> MatrixMath {
        let mut ret: Vec<f64> = vec![0.0; self.vals.len()];
        for i in 0..self.vals.len() {
            ret[i] = -self.vals[i];
        }
        MatrixMath {
            vals: ret,
            original_size: self.original_size,
        }
    }

    fn fill_identity(&mut self) {
        let mut base = 4;
        while base < self.original_size {
            base *= 4;
        }
        let new_square_size = (base as f64).sqrt() as usize;
        let old_square_size = (self.original_size as f64).sqrt() as usize;

        if new_square_size == old_square_size {
            return;
        }

        let size_diff = new_square_size - old_square_size;
        let mut ret: Vec<f64> = vec![0.0; (base * 4).into()];
        for i in 0..new_square_size {
            for j in 0..new_square_size {
                if i >= size_diff && j >= size_diff {
                    ret[i * new_square_size + j] =
                        self.vals[(i - (size_diff)) * old_square_size + j - (size_diff)];
                } else {
                    ret[i * new_square_size + j] = if i == j { 1.0 } else { 0.0 }
                }
            }
        }
        self.vals = ret;
    }

    fn invert_two(&self) -> MatrixMath {
        let mut ret: Vec<f64> = vec![0.0; self.vals.len()];

        let a: f64 = self.vals[0];
        let b: f64 = self.vals[1];
        let c: f64 = self.vals[2];
        let d: f64 = self.vals[3];
        let det: f64 = (a * d) - (b * c);
        ret[0] = d / det;
        ret[1] = -b / det;
        ret[2] = -c / det;
        ret[3] = a / det;
        return MatrixMath {
            vals: ret,
            original_size: self.original_size,
        };
    }

    fn trim_identity(&mut self) {
        let mut base = 4;
        while base < self.vals.len() {
            base *= 4;
        }
        let new_square_size = (self.original_size as f64).sqrt() as usize;
        let old_square_size = (base as f64).sqrt() as usize;

        if new_square_size == old_square_size {
            return;
        }

        let size_diff = old_square_size - new_square_size;
        let mut ret: Vec<f64> = vec![0.0; (base * 4).into()];
        for i in 0..new_square_size {
            for j in 0..new_square_size {
                ret[i * new_square_size + j] =
                    self.vals[(i + size_diff) * old_square_size + j + (size_diff)];
            }
        }
        self.vals = ret;
    }

    fn invert_inner(&self) -> MatrixMath {
        if self.vals.len() == 4 {
            return self.invert_two();
        }

        let half_square: usize = ((self.vals.len() as f64).sqrt() / 2.0) as usize;
        let quarter_count: usize = self.vals.len() / 4;
        //fill and invert e
        let mut e: Vec<f64> = vec![0.0; quarter_count];
        for i in 0..half_square {
            for j in 0..half_square {
                e[i * half_square + j] = self.vals[i * (half_square * 2) + j];
            }
        }

        let e_obj: MatrixMath = MatrixMath {
            vals: e,
            original_size: self.original_size,
        };
        let e_inv: MatrixMath = e_obj.invert_inner();

        //fill the other submatrices
        let mut f_vec: Vec<f64> = vec![0.0; quarter_count];
        let mut g_vec: Vec<f64> = vec![0.0; quarter_count];
        let mut h_vec: Vec<f64> = vec![0.0; quarter_count];
        for i in 0..2 * half_square {
            for j in 0..2 * half_square {
                if i >= half_square || j >= half_square {
                    if i < half_square {
                        f_vec[i * half_square + (j - half_square)] =
                            self.vals[i * (half_square * 2) + j];
                    } else {
                        if j < half_square {
                            g_vec[(i - half_square) * half_square + j] =
                                self.vals[i * (half_square * 2) + j];
                        } else {
                            h_vec[(i - half_square) * half_square + j - half_square] =
                                self.vals[i * (half_square * 2) + j];
                        }
                    }
                }
            }
        }

        let f = MatrixMath {
            vals: f_vec,
            original_size: self.original_size,
        };
        let g = MatrixMath {
            vals: g_vec,
            original_size: self.original_size,
        };
        let h = MatrixMath {
            vals: h_vec,
            original_size: self.original_size,
        };

        //inv hgef
        let hgef_obj: MatrixMath = h - g.clone() * e_inv.clone() * f.clone();
        let inv_hgef: MatrixMath = hgef_obj.invert_inner();

        let e_inv_f = e_inv.clone() * f;
        let g_e_inv = g * e_inv.clone();

        quarters_merge(
            e_inv + e_inv_f.clone() * inv_hgef.clone() * g_e_inv.clone(),
            (e_inv_f * inv_hgef.clone()).sign_swap(),
            (inv_hgef.clone() * g_e_inv).sign_swap(),
            inv_hgef,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn is_testing() {
        assert!(true);
    }

    fn float_equal(one: f64, two: f64) -> bool {
        match (one, two) {
            (a, b) if (a as f32) == (b as f32) => true,
            (a, b) if a == -0.0 || b == -0.0 => {
                if a + b == 0.0 || -(a + b) == 0.0 {
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    #[test]
    fn test_2_x_2() {
        let vals = vec![4.0, 7.0, 2.0, 6.0];
        let inv_given = vec![0.6, -0.7, -0.2, 0.4];
        let inv = invert(vals);
        for i in 0..inv_given.len() {
            assert!(float_equal(inv[i], inv_given[i]))
        }
    }

    fn generate_identity(size: usize) -> Vec<f64> {
        let new_square_size = (size as f64).sqrt() as usize;
        let mut ret: Vec<f64> = vec![0.0; size];
        for i in 0..new_square_size {
            for j in 0..new_square_size {
                if i == j {
                    ret[i * new_square_size + j] = 1.0
                }
            }
        }
        ret
    }

    #[test]
    fn check_invert_identity() {
        for i in 1..6 {
            let ident = generate_identity(4_usize.pow(i));
            let inv = invert(generate_identity(4_usize.pow(i)));
            for j in 0..ident.len() {
                assert!(float_equal(ident[j], inv[j]))
            }
        }
    }

    #[test]
    fn check_4_x_4() {
        for _ in 1..6 {
            let vals = vec![
                13.0, 17.0, 25.0, 12.0, 19.0, 24.0, 16.0, 21.0, 29.0, 9.0, 3.0, 14.0, 23.0, 27.0,
                20.0, 15.0,
            ];
            let inv_given = vec![
                0.0053046525209266107335,
                -0.053014080851339951963,
                0.043653883589643760947,
                0.029232366491467133877,
                -0.072318473817403153419,
                0.004087989098695736796,
                -0.044675880864317695146,
                0.093829083122445006786,
                0.077331127116994354671,
                -0.029719031860359483483,
                0.0033579910453572123787,
                -0.023392382064758938417,
                0.018931282849912400217,
                0.11355525274154824475,
                0.0090033093245084679753,
                -0.11585880215430536628,
            ];
            println!("{:?}", &vals);
            println!("{:?}", &inv_given);
            let inv = invert(vals);
            for i in 0..inv_given.len() {
                assert!(float_equal(inv[i], inv_given[i]))
            }
        }
    }

    #[test]
    fn check_8_x_8() {
        let vals = vec![
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            400000000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            400000000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            400000000.0,
        ];
        let inv_given = vec![
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.001,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.001,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.001,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0000000025,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0000000025,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0000000025,
        ];
        println!("{:?}", &vals);
        println!("{:?}", &inv_given);
        let inv = invert(vals.clone());
        for i in 0..inv_given.len() {
            assert!(float_equal(inv[i], inv_given[i]))
        }
        assert_eq!(inv, inv_given);

        //self * inverse = identity
        let mut l = MatrixMath::new(vals, 64);
        let mut r = MatrixMath::new(inv, 64);
        l.fill_identity();
        r.fill_identity();
        let mut mul = l * r;
        mul.trim_identity();
        let mul = mul.vals;
        let ident = generate_identity(64);
        for i in 0..ident.len() {
            assert!(float_equal(mul[i], ident[i]))
        }
    }
}