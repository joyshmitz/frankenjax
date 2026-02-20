# FJ-P2C-007 Risk Note: FFI Call Interface

## Threat Analysis

| # | Threat | Residual Risk | Mitigation Evidence |
|---|--------|---------------|---------------------|
| 1 | Buffer overflow via FFI callee | Low | Output buffers pre-allocated to exact declared size. Pre-call validation. Security threat matrix T1 |
| 2 | Use-after-free across FFI | Low | Borrows scoped to invoke() unsafe block. Rust borrow checker on safe side. Security threat matrix T2 |
| 3 | Type confusion at boundary | Medium | Explicit dtype-to-C mapping. External function type correctness is caller responsibility. T3 |
| 4 | Code injection via dlopen | Low (V1) | V1: no dynamic loading. All targets statically linked. T4 |
| 5 | Double-free | Negligible | Ownership contract: FrankenJAX owns all buffers. FFI borrows only. T5 |
| 6 | Null pointer dereference | Negligible | Null check at registration time. Non-null invariant maintained. T6 |
| 7 | Segfault in external code | Accepted | Unrecoverable. Process aborts. Documented as known risk. T7 |
| 8 | Thread-safety violation | Medium | FfiRegistry is RwLock-protected. Individual FFI functions assumed thread-safe by contract. T8 |

## Invariant Checklist

| # | Invariant | Status | Evidence |
|---|-----------|--------|----------|
| 1 | FFI calls only through registry | VERIFIED | FfiCall::invoke() resolves target from FfiRegistry.get() |
| 2 | Buffer layout matches declared dtype/shape | VERIFIED | FfiBuffer::new() validates size = product(shape) * dtype_size |
| 3 | Error propagation is structured | VERIFIED | Non-zero return code → FfiError::CallFailed. oracle_error_propagation_is_structured |
| 4 | No dangling pointers after call | VERIFIED | Borrows scoped to unsafe block. invoke_copy_success, invoke_vector_input_output |
| 5 | Memory ownership is one-way | VERIFIED | Input: *const, Output: *mut. No ownership transfer. buffer_lifecycle_borrow_use_return |
| 6 | Registration is unique per name | VERIFIED | registry_duplicate_rejected, adversarial_duplicate_registration |
| 7 | FFI is opaque to transforms | VERIFIED | Contract table p2c007.strict.inv007 |
| 8 | Unsafe confined to invoke() | VERIFIED | #![deny(unsafe_code)] with #[allow(unsafe_code)] on FfiCall::invoke() only |
| 9 | Marshal roundtrip is identity | VERIFIED | metamorphic_marshal_roundtrip_is_identity, prop_f64_scalar_roundtrip (256+ cases) |
| 10 | Thread safety | VERIFIED | adversarial_concurrent_ffi_calls (8 threads) |

## Performance Summary

| Benchmark | Latency | Target | Status |
|-----------|---------|--------|--------|
| ffi_roundtrip/scalar_f64 | 92 ns | < 500 ns | PASS |
| ffi_roundtrip/noop | 40 ns | < 100 ns | PASS |
| ffi_roundtrip/1k_f64_vec | 880 ns | < 2 µs | PASS |
| marshal/scalar_to_buffer | 16 ns | < 100 ns | PASS |
| marshal/buffer_to_scalar | 8 ns | < 100 ns | PASS |
| marshal/1k_tensor_to_buffer | 1.37 µs | < 5 µs | PASS |
| marshal/1k_buffer_to_tensor | 1.59 µs | < 5 µs | PASS |
| registry/lookup | 37 ns | < 100 ns | PASS |

## Test Count

- 49 unit tests (fj-ffi)
- 15 oracle/metamorphic/adversarial tests (ffi_oracle)
- 6 E2E tests (e2e_p2c007)
- **70 total**, all passing
