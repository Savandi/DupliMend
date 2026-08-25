class OnlineOneHotEncoder:
    def __init__(self, max_categories):
        self.max_categories = max_categories
        self.value_to_index = {}
        self.next_index = 0

    def encode(self, values):
        one_hot = [0] * self.max_categories
        if not isinstance(values, list):
            values = [values]

        for value in values:
            if value is None or (isinstance(value, str) and value.strip().lower() in {"", "nan", "null", "none"}):
                continue
            if value not in self.value_to_index:
                if self.next_index < self.max_categories:
                    self.value_to_index[value] = self.next_index
                    self.next_index += 1
                else:
                    continue
            index = self.value_to_index.get(value)
            if index is not None and index < self.max_categories:
                one_hot[index] = 1
        return one_hot

    @property
    def categories(self):
        return [cat for cat, idx in sorted(self.value_to_index.items(), key=lambda x: x[1])]
