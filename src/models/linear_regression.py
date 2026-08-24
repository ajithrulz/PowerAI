from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_linear_regression():

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LinearRegression()
            )
        ]
    )

    return model