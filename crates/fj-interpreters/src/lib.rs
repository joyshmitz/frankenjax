#![forbid(unsafe_code)]

use fj_core::{Atom, Jaxpr, Value, VarId};
use fj_lax::{EvalError, eval_primitive};
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpreterError {
    InputArity {
        expected: usize,
        actual: usize,
    },
    MissingVariable(VarId),
    UnexpectedOutputArity {
        primitive: fj_core::Primitive,
        actual: usize,
    },
    Primitive(EvalError),
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputArity { expected, actual } => {
                write!(
                    f,
                    "input arity mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::MissingVariable(var) => write!(f, "missing variable v{}", var.0),
            Self::UnexpectedOutputArity { primitive, actual } => write!(
                f,
                "expected single-output primitive {}, got {} outputs",
                primitive.as_str(),
                actual
            ),
            Self::Primitive(err) => write!(f, "primitive eval failed: {err}"),
        }
    }
}

impl std::error::Error for InterpreterError {}

impl From<EvalError> for InterpreterError {
    fn from(value: EvalError) -> Self {
        Self::Primitive(value)
    }
}

pub fn eval_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<Value>, InterpreterError> {
    if args.len() != jaxpr.invars.len() {
        return Err(InterpreterError::InputArity {
            expected: jaxpr.invars.len(),
            actual: args.len(),
        });
    }

    let mut env: FxHashMap<VarId, Value> = FxHashMap::default();
    for (idx, var) in jaxpr.invars.iter().enumerate() {
        env.insert(*var, args[idx].clone());
    }

    for eqn in &jaxpr.equations {
        if eqn.outputs.len() != 1 {
            return Err(InterpreterError::UnexpectedOutputArity {
                primitive: eqn.primitive,
                actual: eqn.outputs.len(),
            });
        }

        let mut resolved = Vec::with_capacity(eqn.inputs.len());
        for atom in &eqn.inputs {
            match atom {
                Atom::Var(var) => {
                    let value = env
                        .get(var)
                        .cloned()
                        .ok_or(InterpreterError::MissingVariable(*var))?;
                    resolved.push(value);
                }
                Atom::Lit(lit) => resolved.push(Value::Scalar(*lit)),
            }
        }

        let output = eval_primitive(eqn.primitive, &resolved)?;
        env.insert(eqn.outputs[0], output);
    }

    jaxpr
        .outvars
        .iter()
        .map(|var| {
            env.get(var)
                .cloned()
                .ok_or(InterpreterError::MissingVariable(*var))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{InterpreterError, eval_jaxpr};
    use fj_core::{ProgramSpec, Value, build_program};

    #[test]
    fn eval_simple_add_jaxpr() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let outputs = eval_jaxpr(&jaxpr, &[Value::scalar_i64(4), Value::scalar_i64(5)]);
        assert_eq!(outputs, Ok(vec![Value::scalar_i64(9)]));
    }

    #[test]
    fn eval_vector_add_one_jaxpr() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let output = eval_jaxpr(
            &jaxpr,
            &[Value::vector_i64(&[1, 2, 3]).expect("vector value should build")],
        )
        .expect("vector add should succeed");

        assert_eq!(
            output,
            vec![Value::vector_i64(&[2, 3, 4]).expect("vector value should build")]
        );
    }

    #[test]
    fn input_arity_mismatch_is_reported() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let err = eval_jaxpr(&jaxpr, &[Value::scalar_i64(4)]).expect_err("should fail");
        assert_eq!(
            err,
            InterpreterError::InputArity {
                expected: 2,
                actual: 1,
            }
        );
    }

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn add_commutative(a in -1_000_000i64..1_000_000, b in -1_000_000i64..1_000_000) {
                let jaxpr = build_program(ProgramSpec::Add2);
                let out_ab = eval_jaxpr(&jaxpr, &[Value::scalar_i64(a), Value::scalar_i64(b)])
                    .expect("add should succeed");
                let out_ba = eval_jaxpr(&jaxpr, &[Value::scalar_i64(b), Value::scalar_i64(a)])
                    .expect("add should succeed");
                prop_assert_eq!(out_ab, out_ba);
            }

            #[test]
            fn add_identity(a in -1_000_000i64..1_000_000) {
                let jaxpr = build_program(ProgramSpec::AddOne);
                let result = eval_jaxpr(&jaxpr, &[Value::scalar_i64(a)]);
                prop_assert!(result.is_ok());
            }

            #[test]
            fn reduce_sum_identity_scalar(x in prop::num::f64::NORMAL) {
                use fj_core::{Atom, Equation, Jaxpr, Primitive, VarId};
                use smallvec::smallvec;
                use std::collections::BTreeMap;
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(2)],
                    vec![Equation {
                        primitive: Primitive::ReduceSum,
                        inputs: smallvec![Atom::Var(VarId(1))],
                        outputs: smallvec![VarId(2)],
                        params: BTreeMap::new(),
                    }],
                );
                let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("reduce_sum of scalar should succeed");
                let out_val = out[0].as_f64_scalar().expect("should be scalar");
                prop_assert!((out_val - x).abs() < 1e-10);
            }
        }
    }
}
