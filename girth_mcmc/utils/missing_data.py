from numpy import ma, isin


__all__ = ['tag_missing_data_mcmc']


def tag_missing_data_mcmc(dataset, valid_responses):
    """Checks the data for valid responses.
    
    Args:
        dataset: (array) array to validate
        valid_responses: (array-like) list of valid responses
        
    Returns:
        updated_dataset: (array) data that holds only valid_responses and
                         invalid_fill
    """
    mask = isin(dataset, valid_responses)

    # MCMC uses a masked array to identify missing data
    return ma.masked_array(dataset, ~mask)

