from model_library import fasttext
from keras_model import get_sequences, get_x_y

# fasttext n-gram 生成
# 生成输入文本的n-gram组合词汇
def create_ngram_set(input_list, ngram_value = n_value):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))
# 把生成的各种gram新组合词加入到原先的句子序列中去
def add_ngram(sequences, token_indice, ngram_range = n_value):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences

ngram_set = set()
padded_seqs = get_sequences(train )
for input_list in x_train_padded_seqs:
    for i in range(2, n_value + 1):
        set_of_ngram = create_ngram_set(input_list, ngram_value=i)
        ngram_set.update(set_of_ngram)
start_index = len_vocab + 2
token_idx = {v: k + start_index for k, v in enumerate(ngram_set)} # 给bigram词汇编码
indice_token = {token_indice[k]: k for k in token_indice}
max_words = np.max(list(indice_token.keys())) + 1
word_ids = add_ngram(tokenizer.texts_to_sequences(train_data), token_idx, n_value)
padded_seqs = pad_sequences(word_ids, maxlen = max_length)

return padded_seqs, max_words


if __name__ == '__main__':
    max_words = 100
    n_value = 2

    comp = load_data('comp_reviews_word.csv')
    non = load_data('non_comp_reviews_word.csv')
    hidden = load_data('hidden_reviews_word.csv')
    not_hidden = load_data('not_hidden_reviews_word.csv')
    print('text data load succeed')
    dataset = [comp, non]
    x, y = get_x_y(dataset)