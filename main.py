from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np


def ternary(x):
    return np.base_repr(int(x), 3, padding=5)[-5:]


def entropy_expected(column):
    """Expected information gain after try"""
    p = (column.groupby(column).count() / len(column))
    return -(p * np.log2(p)).sum()


def apply_guess(res, real_word, guess, guesses):
    guesses.append(guess)
    return res[res[guess] == res.loc[real_word, guess]]


def get_best(res, real_word, guesses):
    guess = res.apply(entropy_expected, axis=0).sort_values().index[-1]
    res = apply_guess(res, real_word, guess, guesses)
    return guess, res


def make_res(targetlist_orig, guesslist_orig):
    """Make table where for every possible guess and every possible answer result is calculated"""
    alphabet = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11,
                'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21,
                'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}

    targetlist = pd.DataFrame([list(i)[::-1] for i in targetlist_orig])
    guesslist = pd.DataFrame([list(i)[::-1] for i in guesslist_orig])

    for i in range(5):
        targetlist[i] = targetlist[i].map(alphabet)
        guesslist[i] = guesslist[i].map(alphabet)

    res = np.zeros((len(targetlist), len(guesslist)))
    for guessletter in range(5):
        not_black = np.zeros((len(targetlist), len(guesslist)))
        green = np.zeros((len(targetlist), len(guesslist)))
        for targetletter in range(5):
            d = 1 - cdist(targetlist[[targetletter]], guesslist[[guessletter]], 'hamming')
            not_black += d
            if guessletter == targetletter:
                green = d
        res = res + (2 * 3 ** guessletter) * green + (3 ** guessletter) * ((not_black != 0) & (green == 0))
    res = pd.DataFrame(res, columns=guesslist_orig, index=targetlist_orig)
    return res


def check_target_full_split(res):
    """If there is a word, which has unique response for each of possible words, we try it and don't work with entropy
    If there is such word from list of possible answers its even better (we have a chance to guess word)"""
    nunique = res.nunique(axis=0)
    res_target_only_nunique = nunique[res.index]
    if max(res_target_only_nunique) == len(res):
        return res_target_only_nunique.sort_values().index[-1]
    elif max(nunique) == len(res):
        return nunique.sort_values().index[-1]
    else:
        return None


def find_guess_chain(res, real_word, first_guess):
    """Find series of try-words, optimal for one guessed word"""
    guesses = []
    res = apply_guess(res, real_word, first_guess, guesses)
    while True:
        if len(res) == 1:
            return guesses + list(res.index)
        guess = check_target_full_split(res)
        if guess:
            res = apply_guess(res, real_word, guess, guesses)
        else:
            guess, res = get_best(res, real_word, guesses)
        if guess == real_word:
            return guesses


def split_or_not(res):
    """Split dataframe for parts which can be solwed by two guesses
    To avoid repeatable calculations"""
    queue = [[[], res]]
    splitted = []

    while len(queue):
        part = queue[0][1]
        prev_guesses = queue[0][0]
        next_guess = check_target_full_split(part)
        if next_guess is None:
            guess = part.apply(entropy_expected, axis=0).sort_values().index[-1]
            prev_guesses = prev_guesses + [guess]
            for guess_val in part[guess].unique():
                part2 = part[part[guess] == guess_val]
                queue.append([prev_guesses, part2])
        else:
            splitted.append([part, prev_guesses, next_guess])
        queue = queue[1:]
    return splitted


guesslist_orig = pd.read_csv('guesslist_orig.csv')['0'].tolist()
targetlist_orig = pd.read_csv('targetlist_orig.csv')['0'].tolist()

print('{} words it total, {} can be answers'.format(len(guesslist_orig), len(targetlist_orig)))

res = make_res(targetlist_orig, guesslist_orig)
guesses = []
n = 0
for part, pre_guesses, next_guess in split_or_not(res):
    print(n + len(part), end=' ')
    n += len(part)
    for w in part.index:
        if next_guess == w:
            guesses.append([w, pre_guesses + [next_guess], len(pre_guesses) + 1])
        else:
            g = find_guess_chain(part, w, next_guess)
            guesses.append([w,  pre_guesses + g, len(g) + len(pre_guesses)])

guesses = pd.DataFrame(guesses, columns=['answer', 'guesses', 'guesses_num'])
guesses.to_csv('res.csv', index=False)
