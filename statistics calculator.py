import math
from scipy.stats import levene


class Statistics:
    """
    A class to perform basic statistical calculations on a dataset.

    Attributes:
        data (list): The dataset for statistical analysis.
    """

    def __init__(self, data):
        """
        Initialize the Statistics class with a dataset.

        Args:
            data (list): A list of numerical values.

        Raises:
            ValueError: If the dataset contains fewer than 2 values.
        """
        if not data or len(data) < 2:
            raise ValueError("Data must contain at least 2 values")
        self.data = data

    def mean(self):
        """
        Calculate the mean of the dataset.

        Returns:
            float: The mean of the dataset.
        """
        return sum(self.data) / len(self.data)

    def variance(self):
        """
        Calculate the variance of the dataset.

        Returns:
            float: The variance of the dataset.
        """
        mean = self.mean()
        return sum((x - mean) ** 2 for x in self.data) / (len(self.data) - 1)

    def standard_deviation(self):
        """
        Calculate the standard deviation of the dataset.

        Returns:
            float: The standard deviation of the dataset.
        """
        return math.sqrt(self.variance())


class TTest:
    """
    A class to perform an independent sample t-test.

    Attributes:
        sample1 (Statistics): Statistical object for the first sample.
        sample2 (Statistics): Statistical object for the second sample.
    """

    def __init__(self, sample1, sample2):
        """
        Initialize the TTest class with two datasets.

        Args:
            sample1 (list): First sample dataset.
            sample2 (list): Second sample dataset.
        """
        self.sample1 = Statistics(sample1)
        self.sample2 = Statistics(sample2)

    def levene_test(self):
        """
        Perform Levene's test to check for equality of variances.

        Returns:
            bool: True if variances are equal, False otherwise.
        """
        _, p_value = levene(self.sample1.data, self.sample2.data)
        return p_value > 0.05

    def calculate(self):
        """
        Calculate the t-statistic and degrees of freedom for the two samples.

        Returns:
            tuple: (t-statistic, degrees of freedom, variance of sample1, variance of sample2)

        Raises:
            ValueError: If difference in sample size exceeds 10%.
        """
        n1 = len(self.sample1.data)
        n2 = len(self.sample2.data)
        if abs(n1 / n2 - 1) > 0.1:
            print("Warning: Difference in sample size exceeds 10%, adjust your data for more reliable results")
        mean1 = self.sample1.mean()
        mean2 = self.sample2.mean()
        variance1 = self.sample1.variance()
        variance2 = self.sample2.variance()
        
        #computes the pooled standard errors and degree of freedom based on the outcome of the levene_test
        if self.levene_test():
            pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
            pooled_se = math.sqrt(pooled_variance * (1 / n1 + 1 / n2))
            degree_of_freedom = (n1 + n2) - 2
        else:
            pooled_se = math.sqrt((variance1 / n1) + (variance2 / n2))
            df_numerator = ((variance1 / n1) + (variance2 / n2))
            df_denominator = ((variance1 / n1) ** 2 / (n1 - 1)) + ((variance2 / n2) ** 2 / (n2 - 1))
            degree_of_freedom = df_numerator / df_denominator

        t_statistic = (mean1 - mean2) / pooled_se
        return t_statistic, degree_of_freedom, variance1, variance2


class Correlation:
    """
    A class to calculate Pearson's correlation coefficient between two variables.

    Attributes:
        X (list): First variable dataset.
        Y (list): Second variable dataset.
    """

    def __init__(self, X, Y):
        """
        Initialize the Correlation class with two datasets.

        Args:
            X (list): First variable dataset.
            Y (list): Second variable dataset.

        Raises:
            ValueError: If the two datasets do not have the same length.
        """
        if len(X) != len(Y):
            raise ValueError("Both variables must contain the same number of values")
        self.X = X
        self.Y = Y

    def calculate(self):
        """
        Calculate Pearson's correlation coefficient.

        Returns:
            float: Pearson's correlation coefficient.

        Raises:
            ZeroDivisionError: If the denominator in the calculation is zero.
        """
        X, Y = self.X, self.Y
        n = len(X)
        sum_X = sum(X)
        sum_Y = sum(Y)
        sum_XY = sum(x * y for x, y in zip(X, Y))
        sum_X_squared = sum(x ** 2 for x in X)
        sum_Y_squared = sum(y ** 2 for y in Y)
        r_nominator = n * sum_XY - (sum_X * sum_Y)
        r_denominator = math.sqrt((n * sum_X_squared - sum_X ** 2) * (n * sum_Y_squared - sum_Y ** 2))

        if r_denominator == 0:
            raise ZeroDivisionError("Denominator cannot be zero")

        r = r_nominator / r_denominator
        return r


def get_sample_input(sample_name):
    """
    Prompt the user to input a dataset.

    Args:
        sample_name (str): The name of the dataset (e.g., 'Sample 1').

    Returns:
        list: A list of numerical values input by the user.

    Raises:
        ValueError: If the input contains non-numerical values.
    """
    while True:
        try:
            sample = input(f"enter numbers for {sample_name}, separated by commas:")
            return [float(num.strip()) for num in sample.split(",")]
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")


def main():
    """
    Main function to interact with the user and perform statistical calculations.

    Provides options to perform an independent sample t-test or calculate Pearson's correlation coefficient.
    """
    print("Welcome to the Statistics Calculator")
    while True:
        print("\nOptions")
        print("1: Independent Sample T Test Calculator")
        print("2: Pearson's Correlation Coefficient Calculator")
        print("3: Exit")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice == "1":
            try:
                print("\nIndependent Sample T-Test Calculator")
                sample1 = get_sample_input("Sample 1")
                sample2 = get_sample_input("Sample 2")
                ttest = TTest(sample1, sample2)
                t_stat, df, v1, v2 = ttest.calculate()
                print("\nResults:")
                print(f"T Statistic: {t_stat:.4f}")
                print(f"Degree of Freedom: {df:.2f}")
                print(f"Variance of Sample 1: {v1:.4f}")
                print(f"Variance of Sample 2: {v2:.4f}")
            except ValueError as e:
                print(f"Error: {e}")
        elif choice == "2":
            try:
                print("\nPearson's Correlation Coefficient Calculator")
                first_variable = get_sample_input("X (First Variable)")
                second_variable = get_sample_input("Y (Second Variable)")
                correlation = Correlation(first_variable, second_variable)
                coefficient = correlation.calculate()
                print(f"\nThe Correlation Coefficient (r) is: {coefficient:.4f}")
            except ValueError as e:
                print(f"Error: {e}")
            except ZeroDivisionError as e:
                print(f"Error: {e}")
        elif choice == "3":
            print("Thank you for using the Statistics Calculator. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
 