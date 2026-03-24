use wasm_bindgen::prelude::*;

use alf_nt::alf_nt::AlfNt;
use alf_nt::bigint::M192i;
use alf_nt::ktm::Ktm;

// Global cipher state (single-threaded WASM)
static mut STATE: Option<BenchState> = None;

struct BenchState {
    alf: AlfNt,
    ktm: Ktm,
    data: [u8; 32],
    tweak: [u8; 16],
}

const REF_KEY: [u8; 16] = [
    0x05, 0x0a, 0x0f, 0x14, 0x19, 0x1e, 0x23, 0x28,
    0x2d, 0x32, 0x37, 0x3c, 0x41, 0x46, 0x4b, 0x50,
];
const REF_TWEAK: [u8; 16] = [
    0x0b, 0x16, 0x21, 0x2c, 0x37, 0x42, 0x4d, 0x58,
    0x63, 0x6e, 0x79, 0x84, 0x8f, 0x9a, 0xa5, 0xb0,
];
const REF_APP_ID: u64 = 0xf1f2f3f4f5f6f7f8u64;

/// Run correctness tests (T4 + T5 + ALF-2-7b). Returns true if all pass.
#[wasm_bindgen]
pub fn verify() -> bool {
    unsafe { verify_inner() }
}

unsafe fn verify_inner() -> bool {
    // T4: ALF-2-0b
    let mut qmax = M192i::set_pwr2(16);
    qmax.subc(1);
    let mut alf = AlfNt::new();
    alf.engine_init(qmax, 0);
    let mut ktm = Ktm::new();
    alf.key_init(&mut ktm, &REF_KEY, REF_APP_ID);
    alf.tweak_init(&ktm, &REF_TWEAK);

    let mut d = [0u8; 32];
    alf.encrypt(&mut d);
    if u16::from_le_bytes([d[0], d[1]]) != 0x7e59 { return false; }

    for _ in 0..4 { alf.encrypt(&mut d); }
    if u16::from_le_bytes([d[0], d[1]]) != 0x7501 { return false; }

    for _ in 0..994 { alf.encrypt(&mut d); }
    if u16::from_le_bytes([d[0], d[1]]) != 0x98a9 { return false; }

    alf.prepare_decrypt();
    for _ in 0..999 { alf.decrypt(&mut d); }
    if u16::from_le_bytes([d[0], d[1]]) != 0 { return false; }

    // T5: ALF-2-0n
    let mut qmax5 = M192i::set_pwr2(15);
    qmax5.addc(1);
    qmax5.subc(1);
    let mut alf5 = AlfNt::new();
    alf5.engine_init(qmax5, 0);
    let mut ktm5 = Ktm::new();
    alf5.key_init(&mut ktm5, &REF_KEY, REF_APP_ID);
    alf5.tweak_init(&ktm5, &REF_TWEAK);

    let mut d5 = [0u8; 32];
    alf5.encrypt(&mut d5);
    if u16::from_le_bytes([d5[0], d5[1]]) != 0x7acd { return false; }

    // ALF-2-7b
    let mut qmax7 = M192i::set_pwr2(23);
    qmax7.subc(1);
    let mut alf7 = AlfNt::new();
    alf7.engine_init(qmax7, 0);
    let mut ktm7 = Ktm::new();
    alf7.key_init(&mut ktm7, &REF_KEY, REF_APP_ID);
    alf7.tweak_init(&ktm7, &REF_TWEAK);

    let mut d7 = [0u8; 32];
    alf7.encrypt(&mut d7);
    if d7[0] != 0xdb || d7[1] != 0xfb || d7[2] != 0x08 { return false; }

    alf7.prepare_decrypt();
    alf7.decrypt(&mut d7);
    if d7[0] != 0 || d7[1] != 0 || d7[2] != 0 { return false; }

    true
}

/// Initialize global state for a benchmark variant.
/// 0=ALF-2-0b, 1=ALF-2-7b, 2=ALF-4-0b, 3=ALF-2-0n(Q=50000), 4=ALF-8-0b
#[wasm_bindgen]
pub fn init_variant(id: u32) {
    unsafe { init_variant_inner(id); }
}

unsafe fn init_variant_inner(id: u32) {
    let qmax = match id {
        0 => { let mut q = M192i::set_pwr2(16); q.subc(1); q }
        1 => { let mut q = M192i::set_pwr2(23); q.subc(1); q }
        2 => { let mut q = M192i::set_pwr2(32); q.subc(1); q }
        3 => M192i::set1(49999),
        4 => { let mut q = M192i::set_pwr2(64); q.subc(1); q }
        _ => return,
    };

    let mut alf = AlfNt::new();
    alf.engine_init(qmax, 0);
    let mut ktm = Ktm::new();
    alf.key_init(&mut ktm, &REF_KEY, REF_APP_ID);
    alf.tweak_init(&ktm, &REF_TWEAK);

    STATE = Some(BenchState {
        alf,
        ktm,
        data: [0u8; 32],
        tweak: REF_TWEAK,
    });
}

/// Encrypt n times using global state. Call init_variant first.
#[wasm_bindgen]
pub fn encrypt_n(n: u32) {
    unsafe {
        if let Some(ref mut s) = STATE {
            for _ in 0..n {
                s.alf.encrypt(&mut s.data);
            }
        }
    }
}

/// Switch global state to decrypt mode.
#[wasm_bindgen]
pub fn prepare_decrypt() {
    unsafe {
        if let Some(ref mut s) = STATE {
            s.alf.prepare_decrypt();
        }
    }
}

/// Re-init tweak for encrypt mode (after prepare_decrypt).
#[wasm_bindgen]
pub fn reinit_encrypt() {
    unsafe {
        if let Some(ref mut s) = STATE {
            s.alf.tweak_init(&s.ktm, &s.tweak);
        }
    }
}

/// Decrypt n times using global state. Call prepare_decrypt first.
#[wasm_bindgen]
pub fn decrypt_n(n: u32) {
    unsafe {
        if let Some(ref mut s) = STATE {
            for _ in 0..n {
                s.alf.decrypt(&mut s.data);
            }
        }
    }
}

/// Return variant info string: "name,n,t,rounds"
#[wasm_bindgen]
pub fn variant_info(id: u32) -> String {
    match id {
        0 => "ALF-2-0b,2,0,20".into(),
        1 => "ALF-2-7b,2,7,28".into(),
        2 => "ALF-4-0b,4,0,14".into(),
        3 => "ALF-2-0n (Q=50000),2,0,20".into(),
        4 => "ALF-8-0b,8,0,12".into(),
        _ => "unknown,0,0,0".into(),
    }
}
