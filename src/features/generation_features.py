import pandas as pd
import numpy as np


def process_generation(generation_df):

    df = generation_df.copy()

    # ==================================================
    # 1. Validate required columns
    # ==================================================

    required_columns = [
        "timestamp",
        "psr_type",
        "value",
        "quantity_unit",
        "zone"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Generation dataset is missing columns: "
            f"{missing_columns}"
        )

    # ==================================================
    # 2. Timestamp
    # ==================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # ==================================================
    # 3. Make sure value is numeric
    # ==================================================

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    # ==================================================
    # 4. Display available PSR types
    # ==================================================

    print("\nAvailable PSR types:")

    print(
        sorted(
            df["psr_type"]
            .dropna()
            .unique()
        )
    )

    # ==================================================
    # 5. Aggregate generation by timestamp + PSR type
    # ==================================================

    generation = (
        df.groupby(
            [
                "timestamp",
                "psr_type"
            ],
            as_index=False
        )["value"]
        .sum()
    )

    # ==================================================
    # 6. Convert long format to wide format
    # ==================================================

    generation_wide = (
        generation
        .pivot(
            index="timestamp",
            columns="psr_type",
            values="value"
        )
        .reset_index()
    )

    # Remove column-axis name
    generation_wide.columns.name = None

    # ==================================================
    # 7. Print resulting columns
    # ==================================================

    print("\nGeneration columns after pivot:")

    print(
        generation_wide.columns.tolist()
    )

    # ==================================================
    # 8. Helper function
    # ==================================================

    def get_column(
        column_name
    ):

        if column_name in generation_wide.columns:

            return (
                generation_wide[column_name]
                .fillna(0)
            )

        print(
            f"Warning: {column_name} "
            f"not available in dataset."
        )

        return pd.Series(
            0.0,
            index=generation_wide.index
        )

    # ==================================================
    # 9. Solar
    # ==================================================

    generation_wide[
        "solar_generation_mw"
    ] = get_column(
        "Solar"
    )

    # ==================================================
    # 10. Wind
    # ==================================================

    wind_columns = [
        "Wind Onshore",
        "Wind Offshore"
    ]

    available_wind_columns = [
        column
        for column in wind_columns
        if column in generation_wide.columns
    ]

    if available_wind_columns:

        generation_wide[
            "wind_generation_mw"
        ] = (
            generation_wide[
                available_wind_columns
            ]
            .sum(axis=1)
        )

    else:

        print(
            "Warning: No wind PSR type "
            "was found in the dataset."
        )

        generation_wide[
            "wind_generation_mw"
        ] = 0.0

    # ==================================================
    # 11. Nuclear
    # ==================================================

    generation_wide[
        "nuclear_generation_mw"
    ] = get_column(
        "Nuclear"
    )

    # ==================================================
    # 12. Gas
    # ==================================================

    gas_columns = [
        "Fossil Gas"
    ]

    available_gas_columns = [
        column
        for column in gas_columns
        if column in generation_wide.columns
    ]

    if available_gas_columns:

        generation_wide[
            "gas_generation_mw"
        ] = (
            generation_wide[
                available_gas_columns
            ]
            .sum(axis=1)
        )

    else:

        generation_wide[
            "gas_generation_mw"
        ] = 0.0

    # ==================================================
    # 13. Coal
    # ==================================================

    coal_columns = [
        "Fossil Hard coal",
        "Fossil Coal-derived gas"
    ]

    available_coal_columns = [
        column
        for column in coal_columns
        if column in generation_wide.columns
    ]

    if available_coal_columns:

        generation_wide[
            "coal_generation_mw"
        ] = (
            generation_wide[
                available_coal_columns
            ]
            .sum(axis=1)
        )

    else:

        generation_wide[
            "coal_generation_mw"
        ] = 0.0

    # ==================================================
    # 14. Oil
    # ==================================================

    generation_wide[
        "oil_generation_mw"
    ] = get_column(
        "Fossil Oil"
    )

    # ==================================================
    # 15. Hydro
    # ==================================================

    hydro_columns = [
        "Hydro Pumped Storage",
        "Hydro Run-of-river and poundage",
        "Hydro Water Reservoir"
    ]

    available_hydro_columns = [
        column
        for column in hydro_columns
        if column in generation_wide.columns
    ]

    if available_hydro_columns:

        generation_wide[
            "hydro_generation_mw"
        ] = (
            generation_wide[
                available_hydro_columns
            ]
            .sum(axis=1)
        )

    else:

        generation_wide[
            "hydro_generation_mw"
        ] = 0.0

    # ==================================================
    # 16. Biomass
    # ==================================================

    generation_wide[
        "biomass_generation_mw"
    ] = get_column(
        "Biomass"
    )

    # ==================================================
    # 17. Other renewable
    # ==================================================

    generation_wide[
        "other_renewable_generation_mw"
    ] = get_column(
        "Other renewable"
    )

    # ==================================================
    # 18. Total renewable generation
    # ==================================================

    generation_wide[
        "renewable_generation_mw"
    ] = (
        generation_wide[
            "solar_generation_mw"
        ]
        +
        generation_wide[
            "wind_generation_mw"
        ]
        +
        generation_wide[
            "hydro_generation_mw"
        ]
        +
        generation_wide[
            "biomass_generation_mw"
        ]
        +
        generation_wide[
            "other_renewable_generation_mw"
        ]
    )

    # ==================================================
    # 19. Total generation
    # ==================================================

    generation_wide[
        "total_generation_mw"
    ] = (
        generation_wide[
            "renewable_generation_mw"
        ]
        +
        generation_wide[
            "nuclear_generation_mw"
        ]
        +
        generation_wide[
            "gas_generation_mw"
        ]
        +
        generation_wide[
            "coal_generation_mw"
        ]
        +
        generation_wide[
            "oil_generation_mw"
        ]
    )

    # ==================================================
    # 20. Renewable share
    # ==================================================

    generation_wide[
        "renewable_share"
    ] = np.where(
        generation_wide[
            "total_generation_mw"
        ] > 0,

        generation_wide[
            "renewable_generation_mw"
        ]
        /
        generation_wide[
            "total_generation_mw"
        ],

        np.nan
    )

    # ==================================================
    # 21. Select final columns
    # ==================================================

    final_columns = [
        "timestamp",

        "solar_generation_mw",
        "wind_generation_mw",

        "hydro_generation_mw",
        "biomass_generation_mw",
        "other_renewable_generation_mw",

        "nuclear_generation_mw",

        "gas_generation_mw",
        "coal_generation_mw",
        "oil_generation_mw",

        "renewable_generation_mw",
        "total_generation_mw",
        "renewable_share"
    ]

    generation_wide = (
        generation_wide[
            final_columns
        ]
    )

    # ==================================================
    # 22. Sort
    # ==================================================

    generation_wide = (
        generation_wide
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return generation_wide