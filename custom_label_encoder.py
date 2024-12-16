import numpy as np
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder:
    """
    A custom label encoder that dynamically adds unseen classes during encoding.
    """
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes_ = np.array([])

    def fit(self, classes):
        """
        Fit the encoder with initial classes.
        """
        # Ensure unique and sorted classes
        self.classes_ = np.unique(classes)
        self.encoder.fit(self.classes_)

    def transform(self, values):
        """
        Transform a value or list of values to their encoded representations.
        Dynamically adds unseen values.
        """
        # Ensure values are iterable (convert single value to a list)
        if not isinstance(values, (list, np.ndarray)):
            values = [values]

        # Identify unseen values
        unseen_values = [v for v in values if v not in self.classes_]
        if unseen_values:
            # Add unseen values dynamically
            self.classes_ = np.unique(np.append(self.classes_, unseen_values))
            self.encoder.fit(self.classes_)

        # Transform the values
        return self.encoder.transform(values)

    def fit_transform(self, values):
        """
        Fit and transform a list of values.
        """
        self.fit(values)
        return self.encoder.transform(values)

    def classes(self):
        """
        Return the current classes.
        """
        return self.classes_
