from sklearn.ensemble import RandomForestRegressor


def create_random_forest():

    model = RandomForestRegressor(

        n_estimators=300,

        max_depth=15,

        min_samples_split=5,

        min_samples_leaf=2,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1
    )

    return model