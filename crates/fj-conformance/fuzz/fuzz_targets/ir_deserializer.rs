#![no_main]

mod common;

use fj_core::Jaxpr;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(jaxpr) = serde_json::from_slice::<Jaxpr>(data) {
        let _ = jaxpr.validate_well_formed();
        let _ = jaxpr.canonical_fingerprint();
        let _ = serde_json::to_vec(&jaxpr);
    }
});
