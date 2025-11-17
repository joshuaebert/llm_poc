#include "llama.h"

#include <print>
#include <iostream>
#include <string>
#include <vector>

std::string get_system_prompt() {
    std::string system_header{"<|start_header_id|>system<|end_header_id|>"};

    std::string system_prompt{
        "Heutiges Datum: 7. November 2025"
        " Du bist ein SQL Query ersteller welcher ausschließlichh dafür gemacht ist, SQL queries zu erstellen welche direkt"
        " gegen eine postgresql datenbank abgefeuert werden."};

    std::string table_data{
        "BEGIN TABLE INFO: Produkte (id, name) = id -> id des Produktes, name -> Name des Produktes"
    };

    return system_header + system_prompt + table_data + std::string{"<|eot_id|>"};
}

std::string get_user_message(const std::string &message) {
    std::string user_header{"<|start_header_id|>user<|end_header_id|>"};

    return user_header + message + std::string{"<|eot_id|>"};
}

std::string get_assistant_message() {
    return {"<|start_header_id|>assistant<|end_header_id|>"};
}

std::string get_message(const std::string &message) {
    return get_system_prompt() + get_user_message(message) + get_assistant_message();
}

int main() {
    constexpr auto model_name = "llama.gguf";

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_name, model_params);

    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        llama_backend_free();
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048 * 4;
    ctx_params.n_batch = 512 * 4;

    llama_context *ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const auto vocab = llama_model_get_vocab(model);
    std::cout << "Model loaded! Vocab size: " << llama_vocab_n_tokens(vocab) << std::endl;

    const std::string user_input = get_message("Zeig mir jeden Patienten der in Q4 einen Termin hatte");
    std::vector<llama_token> tokens{};
    std::int32_t n_tokens = -llama_tokenize(vocab, user_input.c_str(), user_input.length(), nullptr, 0, true, true);

    if (n_tokens == std::numeric_limits<std::int32_t>::min()) {
        std::println("Tokenize overflow");
        return 1;
    }

    tokens.resize(n_tokens);
    n_tokens = llama_tokenize(vocab, user_input.c_str(), user_input.length(), tokens.data(), tokens.size(), true,
                              true);

    if (n_tokens <= 0) {
        std::println("Tokenization failed");
        return 1;
    }

    auto batch = llama_batch_init(n_tokens, 0, 1);
    std::int32_t current_pos{0};

    for (std::int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.logits[i] = false;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        ++current_pos;
    }

    batch.n_tokens = n_tokens;
    batch.logits[n_tokens - 1] = true;

    auto decode_result = llama_decode(ctx, batch);

    if (decode_result < 0) {
        std::println("Fatal decode error...");
        return 1;
    }

    std::vector<llama_token> predicted_tokens{};

    std::int32_t total_generated = 1;

    auto *sampler = llama_sampler_chain_init(
            llama_sampler_chain_default_params()
    );

    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.01f));

    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    while (true) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        predicted_tokens.push_back(token);

        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = current_pos++;
        batch.logits[0] = true;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;

        decode_result = llama_decode(ctx, batch);

        if (decode_result < 0) {
            std::println("Fatal decode error...");
            break;
        }
        ++total_generated;
    }

    std::string result_text{};
    std::int32_t n_chars = llama_detokenize(
            vocab,
            predicted_tokens.data(),
            predicted_tokens.size(),
            nullptr,  // NULL buffer
            0,        // 0 size
            false,    // remove_special
            true      // unparse_special
    );

    if (n_chars > 0) {
        std::println("Failed to detokenize");
        return 1;
    }

    result_text.resize(-n_chars);

    n_chars = llama_detokenize(
            vocab,
            predicted_tokens.data(),
            predicted_tokens.size(),
            result_text.data(),  // Now has proper buffer
            result_text.size(),   // Correct size
            false,
            false
    );


    if (n_chars < 0) {
        std::println("Failed to detokenize");
        return 1;
    }

    std::println("Output length: {}, t:{}", result_text.size(), tokens.size());

    size_t start = result_text.find_first_not_of(" \n\r\t");
    if (start != std::string::npos) {
        result_text = result_text.substr(start);
    }

    std::println("{}", result_text);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}