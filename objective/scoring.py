import tabulate


class ScoreList:

    def __init__(self, sentences, scores):
        self.sentences = sentences
        self.scores = scores

    def sort_by_emotion(self, target_emotion):
        if target_emotion not in self.emotion_classes:
            raise ValueError('Emotion "{}" not supported by this list.')


    def print_top_k_scores(self, target_emotion, k, show_all_emotions=False):
        self.sort_by_emotion(target_emotion)
        if k > len(self.score_list):
            k = len(self.score_list)
        if show_all_emotions:
            table_data = [[" ".join(ss.sentence)] + [ss.scores[emotion] for emotion in self.emotion_classes] for ss in self.score_list[:k]]
            headers = ['Sentence'] + list(self.emotion_classes)
        else:
            table_data = [[" ".join(ss.sentence), ss.scores[target_emotion]] for ss in self.score_list[:k]]
            headers = ['Sentence', target_emotion]
        table = tabulate.tabulate(table_data, headers=headers)
        print(table)


class EmotionScore:
    """ This class holds a sentence with it's associated emotion scores """

    def __init__(self, scores, label_field):
        self.scores = {label_field.vocab.itos[emotion_index]: score for emotion_index, score in enumerate(scores)}