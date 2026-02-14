#![allow(dead_code)]

use fj_core::{
    CompatibilityMode, DType, Literal, Primitive, ProgramSpec, Shape, TensorValue, Transform,
    Value,
};
use std::collections::BTreeMap;

pub struct ByteCursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn take_raw_u8(&mut self) -> u8 {
        if self.data.is_empty() {
            return 0;
        }

        let byte = self.data[self.offset % self.data.len()];
        self.offset = self.offset.saturating_add(1);
        byte
    }

    #[must_use]
    pub fn take_u8(&mut self) -> u8 {
        self.take_raw_u8()
    }

    #[must_use]
    pub fn take_bool(&mut self) -> bool {
        self.take_raw_u8() % 2 == 0
    }

    #[must_use]
    pub fn take_u16(&mut self) -> u16 {
        u16::from(self.take_raw_u8()) | (u16::from(self.take_raw_u8()) << 8)
    }

    #[must_use]
    pub fn take_u32(&mut self) -> u32 {
        u32::from(self.take_raw_u8())
            | (u32::from(self.take_raw_u8()) << 8)
            | (u32::from(self.take_raw_u8()) << 16)
            | (u32::from(self.take_raw_u8()) << 24)
    }

    #[must_use]
    pub fn take_u64(&mut self) -> u64 {
        u64::from(self.take_u32()) | (u64::from(self.take_u32()) << 32)
    }

    #[must_use]
    pub fn take_usize(&mut self, inclusive_max: usize) -> usize {
        if inclusive_max == 0 {
            return 0;
        }
        usize::from(self.take_raw_u8()) % (inclusive_max + 1)
    }

    #[must_use]
    pub fn take_string(&mut self, max_len: usize) -> String {
        let len = self.take_usize(max_len);
        let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789_-";
        let mut out = String::with_capacity(len);
        for _ in 0..len {
            let idx = usize::from(self.take_raw_u8()) % alphabet.len();
            out.push(char::from(alphabet[idx]));
        }
        out
    }
}

#[must_use]
pub fn sample_mode(cursor: &mut ByteCursor<'_>) -> CompatibilityMode {
    if cursor.take_bool() {
        CompatibilityMode::Strict
    } else {
        CompatibilityMode::Hardened
    }
}

#[must_use]
pub fn sample_program(cursor: &mut ByteCursor<'_>) -> ProgramSpec {
    match cursor.take_u8() % 8 {
        0 => ProgramSpec::Add2,
        1 => ProgramSpec::Square,
        2 => ProgramSpec::SquarePlusLinear,
        3 => ProgramSpec::AddOne,
        4 => ProgramSpec::SinX,
        5 => ProgramSpec::CosX,
        6 => ProgramSpec::Dot3,
        _ => ProgramSpec::ReduceSumVec,
    }
}

#[must_use]
pub fn sample_transform(cursor: &mut ByteCursor<'_>) -> Transform {
    match cursor.take_u8() % 3 {
        0 => Transform::Jit,
        1 => Transform::Grad,
        _ => Transform::Vmap,
    }
}

#[must_use]
pub fn sample_primitive(cursor: &mut ByteCursor<'_>) -> Primitive {
    match cursor.take_u8() % 13 {
        0 => Primitive::Add,
        1 => Primitive::Mul,
        2 => Primitive::Dot,
        3 => Primitive::Sin,
        4 => Primitive::Cos,
        5 => Primitive::ReduceSum,
        6 => Primitive::Reshape,
        7 => Primitive::Slice,
        8 => Primitive::Gather,
        9 => Primitive::Scatter,
        10 => Primitive::Transpose,
        11 => Primitive::BroadcastInDim,
        _ => Primitive::Concatenate,
    }
}

#[must_use]
pub fn primitive_arity(primitive: Primitive) -> usize {
    match primitive {
        Primitive::Add | Primitive::Mul | Primitive::Dot | Primitive::Gather | Primitive::Scatter => 2,
        Primitive::Sin
        | Primitive::Cos
        | Primitive::ReduceSum
        | Primitive::Reshape
        | Primitive::Slice
        | Primitive::Transpose
        | Primitive::BroadcastInDim => 1,
        Primitive::Concatenate => 2,
    }
}

#[must_use]
pub fn sample_dtype(cursor: &mut ByteCursor<'_>) -> DType {
    match cursor.take_u8() % 5 {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::I32,
        3 => DType::I64,
        _ => DType::Bool,
    }
}

#[must_use]
pub fn sample_shape(cursor: &mut ByteCursor<'_>, max_rank: usize, max_dim: u32) -> Shape {
    let rank = cursor.take_usize(max_rank);
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        let limit = max_dim.max(1);
        dims.push(cursor.take_u32() % (limit + 1));
    }
    Shape { dims }
}

#[must_use]
pub fn sample_literal(cursor: &mut ByteCursor<'_>, dtype: DType) -> Literal {
    match dtype {
        DType::Bool => Literal::Bool(cursor.take_bool()),
        DType::I32 => Literal::I64(i64::from((cursor.take_u32() % 10_000) as i32)),
        DType::I64 => Literal::I64((cursor.take_u64() % 1_000_000) as i64),
        DType::F32 | DType::F64 => Literal::F64Bits(cursor.take_u64()),
    }
}

#[must_use]
pub fn sample_value(cursor: &mut ByteCursor<'_>) -> Value {
    if cursor.take_bool() {
        let dtype = sample_dtype(cursor);
        return Value::Scalar(sample_literal(cursor, dtype));
    }

    let dtype = sample_dtype(cursor);
    let shape = sample_shape(cursor, 3, 4);
    let Some(element_count) = shape.element_count() else {
        return Value::Scalar(sample_literal(cursor, dtype));
    };

    let Ok(element_count) = usize::try_from(element_count) else {
        return Value::Scalar(sample_literal(cursor, dtype));
    };

    if element_count > 64 {
        return Value::Scalar(sample_literal(cursor, dtype));
    }

    let mut elements = Vec::with_capacity(element_count);
    for _ in 0..element_count {
        elements.push(sample_literal(cursor, dtype));
    }

    match TensorValue::new(dtype, shape, elements) {
        Ok(tensor) => Value::Tensor(tensor),
        Err(_) => Value::Scalar(sample_literal(cursor, dtype)),
    }
}

#[must_use]
pub fn sample_values(cursor: &mut ByteCursor<'_>, max_len: usize) -> Vec<Value> {
    let len = cursor.take_usize(max_len);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(sample_value(cursor));
    }
    out
}

#[must_use]
pub fn sample_backend(cursor: &mut ByteCursor<'_>) -> String {
    let backend = cursor.take_string(10);
    if backend.trim().is_empty() {
        "cpu".to_owned()
    } else {
        backend
    }
}

#[must_use]
pub fn sample_compile_options(
    cursor: &mut ByteCursor<'_>,
    max_entries: usize,
) -> BTreeMap<String, String> {
    let mut options = BTreeMap::new();
    let entries = cursor.take_usize(max_entries);
    for idx in 0..entries {
        let key = format!("opt{}_{}", idx, cursor.take_string(6));
        let value = cursor.take_string(14);
        options.insert(key, value);
    }
    options
}

#[must_use]
pub fn sample_unknown_features(cursor: &mut ByteCursor<'_>, max_entries: usize) -> Vec<String> {
    let entries = cursor.take_usize(max_entries);
    let mut out = Vec::with_capacity(entries);
    for idx in 0..entries {
        let base = cursor.take_string(12);
        let feature = if base.is_empty() {
            format!("unknown_feature_{}", idx)
        } else {
            base
        };
        out.push(feature);
    }
    out
}

#[must_use]
pub fn sample_evidence_id(cursor: &mut ByteCursor<'_>, index: usize) -> String {
    if cursor.take_u8() % 5 == 0 {
        return String::new();
    }

    let value = cursor.take_string(16);
    if value.is_empty() {
        format!("ev-{}", index)
    } else {
        value
    }
}

#[must_use]
pub fn sample_primitive_params(
    cursor: &mut ByteCursor<'_>,
    primitive: Primitive,
) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();

    match primitive {
        Primitive::ReduceSum => {
            if cursor.take_bool() {
                let axis_count = 1 + cursor.take_usize(2);
                let axes = (0..axis_count)
                    .map(|_| cursor.take_usize(3).to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                params.insert("axes".to_owned(), axes);
            }
        }
        Primitive::Reshape => {
            let rank = 1 + cursor.take_usize(3);
            let shape = (0..rank)
                .map(|_| (1 + cursor.take_usize(4)).to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("new_shape".to_owned(), shape);
        }
        Primitive::Slice => {
            let rank = 1 + cursor.take_usize(3);
            let mut starts = Vec::with_capacity(rank);
            let mut limits = Vec::with_capacity(rank);
            for _ in 0..rank {
                let start = cursor.take_usize(3);
                let width = 1 + cursor.take_usize(3);
                starts.push(start.to_string());
                limits.push((start + width).to_string());
            }
            params.insert("start_indices".to_owned(), starts.join(","));
            params.insert("limit_indices".to_owned(), limits.join(","));
        }
        Primitive::Gather => {
            if cursor.take_bool() {
                let rank = 1 + cursor.take_usize(3);
                let sizes = (0..rank)
                    .map(|_| (1 + cursor.take_usize(3)).to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                params.insert("slice_sizes".to_owned(), sizes);
            }
        }
        Primitive::Transpose => {
            let rank = 1 + cursor.take_usize(3);
            let mut axes = (0..rank).collect::<Vec<_>>();
            if cursor.take_bool() {
                axes.reverse();
            }
            let encoded = axes
                .into_iter()
                .map(|axis| axis.to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("permutation".to_owned(), encoded);
        }
        Primitive::BroadcastInDim => {
            let rank = 1 + cursor.take_usize(4);
            let target_shape = (0..rank)
                .map(|_| (1 + cursor.take_usize(4)).to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("shape".to_owned(), target_shape);

            if cursor.take_bool() {
                let dims = (0..rank)
                    .map(|axis| axis.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                params.insert("broadcast_dimensions".to_owned(), dims);
            }
        }
        Primitive::Concatenate => {
            params.insert("dimension".to_owned(), cursor.take_usize(3).to_string());
        }
        Primitive::Add
        | Primitive::Mul
        | Primitive::Dot
        | Primitive::Sin
        | Primitive::Cos
        | Primitive::Scatter => {}
    }

    params
}
