# Integration tests

> Note: you will require a GPU to complete this integration test, which takes approx 75 minutes to run on a NVIDIA Tesla K80 card and Intel Xeon E5-2690 v3 ([`STANDARD_NC6`](https://docs.microsoft.com/en-us/azure/virtual-machines/nc-series) in Microsoft Azure)

Once you have completed the [getting started](../README.md#getting-started) section of this project and activated your virtual environment, you can execute an end-to-end integration test by running from the main project directory:

```bash
./tests/integration_test.sh
```

This will perform the following actions:

1. Generate synthetic slides
2. Stain the slides
3. Train the classifier on the stained slides
4. Test the classifier and output a ROC curve under `Data/test/roc_image.png`
