import re, unicodedata, string
from token_list import strip_tokens
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Disabled for now, may test in other iterations
# def unicode_to_ascii(seq):
#     return ''.join(c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn')

def process_raw(raw_data):
    raw_data = [seq.lower().strip() for seq in raw_data]

    # Creating a space between a word and the punctuation following it
    # Eg: "he is a boy." => "he is a boy ."
    raw_data = [re.sub(r"([?.!,¿])", r" \1 ", seq) for seq in raw_data]
    raw_data = [re.sub(r'[" "]+', " ", seq) for seq in raw_data]

    # Replacing everything with space except (characters, ".", "?", "!", ",")
    filtered_punctuations = string.punctuation
    exclude = [',', '!', '.', '?']

    for c in filtered_punctuations:
        if c in exclude:
            filtered_punctuations = filtered_punctuations.replace(c, '')

    table = str.maketrans('', '', filtered_punctuations)
    raw_data = [seq.translate(table) for seq in raw_data]
    
    # Append start and end tokens to sequences
    processed_raw = []
    for seq in raw_data:
        words = seq.split()
        words = [word.strip() for word in words]
        processed_raw.append(' '.join(words))

    # processed_raw = ['<s> ' + seq + ' <e>' for seq in processed_raw]
    processed_raw = [seq + ' <e>' for seq in processed_raw]

    return processed_raw

def generate_input(processed_raw):
    output = ''
    for char in processed_raw:
        if char in strip_tokens:
            output += strip_tokens[char]
        else:
            output += char          
    return output

def get_max_len(input_data):
    return max([len(data.split()) for data in input_data])

def tokenize_pad_data(data, pad_len=None):
    tk = Tokenizer(char_level=False, filters='')
    tk.fit_on_texts(data)
    data = tk.texts_to_sequences(data)
    data = pad_sequences(data, padding='post', maxlen=pad_len)
    return data, tk

def process_data(processed_input, processed_target, pad_len=None):    
    tokenized_input, input_tokenizer = tokenize_pad_data(data=processed_input, pad_len=pad_len)
    tokenized_target, target_tokenizer = tokenize_pad_data(data=processed_target, pad_len=pad_len)
    return tokenized_input, input_tokenizer, tokenized_target, target_tokenizer

def convert(tokenizer, tokenized_data):
    original = []
    print('Tokenized Data: {}'.format(tokenized_data))
    print('\n')
    for token in tokenized_data:
        if token != 0:
            if token in tokenizer.index_word:
                original.append(tokenizer.index_word[token])
            else:
                original.append('<unk>')
    print('Original Data: {}'.format(original))