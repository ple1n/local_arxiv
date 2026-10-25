//! Text passage chunking for inspection and indexing.
//!
//! Passages preserve their source byte ranges. Callers can provide their model tokenizer's
//! token cost; the default helper uses a character-based estimate.

/// One chunk of source text.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Chunk {
    /// Normalized text stored for this chunk.
    pub text: String,
    /// Half-open UTF-8 byte range in the original body.
    pub byte_range: (usize, usize),
    /// Whitespace-delimited word count for display.
    pub word_count: usize,
    /// Sum of the caller-supplied costs for the packed units.
    pub cost: usize,
}

struct Unit {
    text: String,
    start: usize,
    end: usize,
    starts_paragraph: bool,
}

/// Split text using a conservative estimate of one token per four UTF-8 bytes.
///
/// Paragraphs are kept intact where possible, then split at sentence and word boundaries.
/// Supply [`chunk_text_with_cost`] with a tokenizer-backed cost function when available;
/// the estimate is not guaranteed to match a model's tokenizer.
#[must_use]
pub fn chunk_text(body: &str, token_budget: usize) -> Vec<Chunk> {
    chunk_text_with_cost(body, token_budget, &estimated_tokens)
}

/// Split text using a caller-supplied cost function, such as an embedding model tokenizer.
#[must_use]
pub fn chunk_text_with_cost(body: &str, budget: usize, cost: &dyn Fn(&str) -> usize) -> Vec<Chunk> {
    if body.is_empty() || budget == 0 {
        return Vec::new();
    }

    let mut packer = Packer {
        body,
        budget,
        cost,
        current: Vec::new(),
        current_cost: 0,
        chunks: Vec::new(),
    };
    for (start, end) in paragraph_ranges(body) {
        let text = normalize(&body[start..end]);
        if !text.is_empty() {
            packer.place(Unit {
                text,
                start,
                end,
                starts_paragraph: true,
            });
        }
    }
    packer.flush();
    packer.chunks
}

struct Packer<'a> {
    body: &'a str,
    budget: usize,
    cost: &'a dyn Fn(&str) -> usize,
    current: Vec<Unit>,
    current_cost: usize,
    chunks: Vec<Chunk>,
}

impl Packer<'_> {
    fn place(&mut self, unit: Unit) {
        let cost = (self.cost)(&unit.text);
        if cost <= self.budget {
            self.append(unit, cost);
            return;
        }

        let sentences = sentence_ranges(self.body, unit.start, unit.end);
        if sentences.len() > 1 {
            for (index, (start, end)) in sentences.into_iter().enumerate() {
                let text = normalize(&self.body[start..end]);
                if !text.is_empty() {
                    self.place(Unit {
                        text,
                        start,
                        end,
                        starts_paragraph: unit.starts_paragraph && index == 0,
                    });
                }
            }
            return;
        }

        let words = word_ranges(self.body, unit.start, unit.end);
        if words.len() > 1 {
            for (index, (start, end)) in words.into_iter().enumerate() {
                self.place(Unit {
                    text: self.body[start..end].to_owned(),
                    start,
                    end,
                    starts_paragraph: unit.starts_paragraph && index == 0,
                });
            }
            return;
        }

        let fragments = fragment_ranges(self.body, unit.start, unit.end, self.budget, self.cost);
        if fragments.len() > 1 {
            for (index, (start, end)) in fragments.into_iter().enumerate() {
                let text = self.body[start..end].to_owned();
                let cost = (self.cost)(&text);
                self.append(
                    Unit {
                        text,
                        start,
                        end,
                        starts_paragraph: unit.starts_paragraph && index == 0,
                    },
                    cost,
                );
            }
        } else {
            self.append(unit, cost);
        }
    }

    fn append(&mut self, unit: Unit, cost: usize) {
        if !self.current.is_empty() && self.current_cost + cost > self.budget {
            self.flush();
        }
        self.current.push(unit);
        self.current_cost += cost;
    }

    fn flush(&mut self) {
        let (Some(first), Some(last)) = (self.current.first(), self.current.last()) else {
            return;
        };
        let mut text = String::new();
        for unit in &self.current {
            if !text.is_empty() {
                text.push(if unit.starts_paragraph { '\n' } else { ' ' });
            }
            text.push_str(&unit.text);
        }
        self.chunks.push(Chunk {
            word_count: word_count(&text),
            cost: self.current_cost,
            text,
            byte_range: (first.start, last.end),
        });
        self.current.clear();
        self.current_cost = 0;
    }
}

fn paragraph_ranges(body: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut cursor = 0;
    while cursor < body.len() {
        if body.as_bytes()[cursor] == b'\n' {
            let mut next = cursor + 1;
            while next < body.len() && matches!(body.as_bytes()[next], b' ' | b'\t' | b'\r') {
                next += 1;
            }
            if next < body.len() && body.as_bytes()[next] == b'\n' {
                push_trimmed_range(body, start, cursor, &mut ranges);
                start = next + 1;
                cursor = start;
                continue;
            }
        }
        cursor += body[cursor..].chars().next().map_or(1, char::len_utf8);
    }
    push_trimmed_range(body, start, body.len(), &mut ranges);
    ranges
}

fn push_trimmed_range(body: &str, start: usize, end: usize, out: &mut Vec<(usize, usize)>) {
    let slice = &body[start..end];
    let leading = slice.len() - slice.trim_start().len();
    let trimmed = slice.trim();
    if !trimmed.is_empty() {
        out.push((start + leading, start + leading + trimmed.len()));
    }
}

fn sentence_ranges(body: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut sentence_start = start;
    let mut cursor = start;
    while cursor < end {
        let ch = body[cursor..].chars().next().unwrap_or_default();
        let next = cursor + ch.len_utf8();
        if matches!(ch, '.' | '!' | '?') && next < end {
            let following = body[next..].chars().next().unwrap_or_default();
            if following.is_whitespace() {
                push_trimmed_range(body, sentence_start, next, &mut ranges);
                sentence_start = next;
            }
        }
        cursor = next;
    }
    push_trimmed_range(body, sentence_start, end, &mut ranges);
    ranges
}

fn word_ranges(body: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut word_start = None;
    for (offset, ch) in body[start..end].char_indices() {
        let absolute = start + offset;
        if ch.is_whitespace() {
            if let Some(word_start) = word_start.take() {
                ranges.push((word_start, absolute));
            }
        } else if word_start.is_none() {
            word_start = Some(absolute);
        }
    }
    if let Some(word_start) = word_start {
        ranges.push((word_start, end));
    }
    ranges
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Approximate tokenizer cost. Model tokenizers vary, so callers should use their provider's
/// tokenizer where possible and leave context headroom when relying on this estimate.
#[must_use]
pub fn estimated_tokens(text: &str) -> usize {
    text.len().div_ceil(4).max(1)
}

fn fragment_ranges(
    body: &str,
    start: usize,
    end: usize,
    budget: usize,
    cost: &dyn Fn(&str) -> usize,
) -> Vec<(usize, usize)> {
    if cost(&body[start..end]) <= budget || body[start..end].chars().count() <= 1 {
        return vec![(start, end)];
    }
    let middle = start + (end - start) / 2;
    let split = (middle..end)
        .find(|&index| body.is_char_boundary(index))
        .or_else(|| {
            (start..middle)
                .rev()
                .find(|&index| body.is_char_boundary(index))
        });
    let Some(split) = split.filter(|split| *split > start && *split < end) else {
        return vec![(start, end)];
    };
    let mut ranges = fragment_ranges(body, start, split, budget, cost);
    ranges.extend(fragment_ranges(body, split, end, budget, cost));
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_paragraphs_together_and_tracks_source_ranges() {
        let body = "First small paragraph.\n\nSecond paragraph here.";
        let chunks = chunk_text_with_cost(body, 4, &word_count);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "First small paragraph.");
        assert_eq!(
            &body[chunks[0].byte_range.0..chunks[0].byte_range.1],
            "First small paragraph."
        );
        assert_eq!(chunks[1].text, "Second paragraph here.");
    }

    #[test]
    fn splits_long_paragraphs_at_sentence_then_word_boundaries() {
        let chunks = chunk_text_with_cost("One two three. Four five six.", 3, &word_count);
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<Vec<_>>(),
            ["One two three.", "Four five six."]
        );
        assert!(chunks.iter().all(|chunk| chunk.word_count <= 3));
    }

    #[test]
    fn zero_budget_produces_no_chunks() {
        assert!(chunk_text("some text", 0).is_empty());
    }

    #[test]
    fn handles_unicode_ranges() {
        let body = "αβγ δ";
        let chunks = chunk_text_with_cost(body, 1, &word_count);
        assert_eq!(chunks[0].text, "αβγ");
        assert_eq!(&body[chunks[0].byte_range.0..chunks[0].byte_range.1], "αβγ");
    }

    #[test]
    fn custom_cost_splits_long_units_and_respects_budget() {
        let body = "abcdefghij";
        let chunks = chunk_text_with_cost(body, 3, &|text| text.len());
        assert!(chunks.len() >= 4);
        assert!(chunks.iter().all(|chunk| chunk.cost <= 3));
        let reconstructed: String = chunks
            .iter()
            .flat_map(|chunk| chunk.text.chars())
            .filter(|character| !character.is_whitespace())
            .collect();
        assert_eq!(reconstructed, body);
    }
}
