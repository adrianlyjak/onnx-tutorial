export class SimpleTokenizer {
  constructor(config) {
    this.vocab = config.vocab;
    this.special_tokens = config.special_tokens;
    this.max_length = config.max_length;
    this.padding_side = config.padding_side;

    // Create reverse vocab (id -> token)
    this.ids_to_tokens = Object.fromEntries(
      Object.entries(this.vocab).map(([token, id]) => [id, token])
    );
  }

  tokenize(text) {
    // Basic whitespace tokenization - you might want to enhance this
    // based on your specific needs
    const tokens = text.toLowerCase().split(/\s+/);

    // Convert to ids
    const input_ids = tokens.map(
      (token) => this.vocab[token] || this.vocab[this.special_tokens.unk_token]
    );

    // Add special tokens
    input_ids.unshift(this.vocab[this.special_tokens.cls_token]);
    input_ids.push(this.vocab[this.special_tokens.sep_token]);

    // Padding if needed
    while (input_ids.length < this.max_length) {
      input_ids.push(this.vocab[this.special_tokens.pad_token]);
    }

    // Truncate if needed
    if (input_ids.length > this.max_length) {
      input_ids.length = this.max_length;
    }

    return input_ids;
  }

  decode(ids) {
    return ids
      .map((id) => this.ids_to_tokens[id] || this.special_tokens.unk_token)
      .join(" ");
  }
}
