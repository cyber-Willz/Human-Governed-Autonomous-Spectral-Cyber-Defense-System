//! Minimal chat templating for the two special-token families this crate targets.
//!
//! This is **not** a Jinja engine. It hard-codes the standard single-turn/multi-turn layouts
//! for ChatML (Qwen2.5 / Qwen3) and Llama-3, chosen by which special tokens the tokenizer
//! defines. Checkpoints with bespoke templates (tool calling, custom system prompts injected
//! by the template, Mistral `[INST]`, …) need their own rendering — pass a raw prompt instead.

use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// `<|im_start|>role\ncontent<|im_end|>\n` (Qwen2.5, Qwen3, many fine-tunes).
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` (Llama 3.x).
    Llama3,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Self { role: role.to_owned(), content: content.to_owned() }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    /// Qwen3 only: append an empty `<think>\n\n</think>\n\n` block after the assistant header,
    /// which is what its template does for `enable_thinking=False`.
    pub no_think: bool,
}

pub fn detect_format(tok: &Tokenizer) -> Option<ChatFormat> {
    if tok.token_to_id("<|im_start|>").is_some() && tok.token_to_id("<|im_end|>").is_some() {
        Some(ChatFormat::ChatMl)
    } else if tok.token_to_id("<|start_header_id|>").is_some() && tok.token_to_id("<|eot_id|>").is_some() {
        Some(ChatFormat::Llama3)
    } else {
        None
    }
}

/// Renders `messages` and appends the assistant generation prompt. Tokenize the result with
/// `add_special_tokens = false` (the template already carries every special token, including
/// Llama-3's `<|begin_of_text|>`).
pub fn render(format: ChatFormat, messages: &[Message], opts: &ChatOptions) -> String {
    let mut out = String::new();
    match format {
        ChatFormat::ChatMl => {
            for m in messages {
                out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
            }
            out.push_str("<|im_start|>assistant\n");
            if opts.no_think {
                out.push_str("<think>\n\n</think>\n\n");
            }
        }
        ChatFormat::Llama3 => {
            out.push_str("<|begin_of_text|>");
            for m in messages {
                out.push_str(&format!(
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    m.role, m.content
                ));
            }
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::test_support::word_level_json;

    #[test]
    fn chatml_layout() {
        let msgs = [Message::new("system", "Be brief."), Message::new("user", "Hi")];
        assert_eq!(
            render(ChatFormat::ChatMl, &msgs, &ChatOptions::default()),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(render(ChatFormat::ChatMl, &msgs, &ChatOptions { no_think: true })
            .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn llama3_layout() {
        let msgs = [Message::new("user", "Hi")];
        assert_eq!(
            render(ChatFormat::Llama3, &msgs, &ChatOptions::default()),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn detection_follows_the_tokenizer_vocab() {
        let tok = Tokenizer::from_json_str(&word_level_json()).unwrap();
        assert_eq!(detect_format(&tok), Some(ChatFormat::ChatMl)); // fixture defines <|im_*|>
    }
}
