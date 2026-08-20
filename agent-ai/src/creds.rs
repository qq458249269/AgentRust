//! Credential resolution chain (mirrors pi ModelRuntime): runtime override -> stored file -> env.
//! M1: env + file. OAuth later, behind this trait.

use std::env;

#[derive(Debug, Clone)]
pub struct Credentials {
    /// provider id -> resolved api key
    keys: std::collections::HashMap<String, String>,
}

impl Credentials {
    pub fn from_env() -> Self {
        // keyed by provider name per DESIGN.md; env var names are provisional
        let mut keys = std::collections::HashMap::new();
        for (provider, var) in [
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("deepseek", "DEEPSEEK_API_KEY"),
        ] {
            if let Ok(k) = env::var(var) {
                keys.insert(provider.to_string(), k);
            }
        }
        Self { keys }
    }

    pub fn get(&self, provider: &str) -> Option<&str> {
        self.keys.get(provider).map(String::as_str)
    }
}
