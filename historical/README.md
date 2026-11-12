# Original submission

The notebook and PDF in this directory retain the exact bytes from the original repository. The saved notebook outputs and the report preserve the historical experiment and its figures.

The corrected forecasting entry point is the [current notebook](../Demand_forecasting.ipynb). Use the [report with correction notice](../final_report.pdf) and the [validation report](../docs/forecasting-validation.md) for the amended methodology and results. The original 4.80% Prophet MAPE was calculated by pairing Sunday predictions with Friday sales by row position. Historical full-series imputation also used later observations.

To inspect the historical notebook interactively, start Jupyter from the repository root. To rerun its legacy cells, use a kernel with the original analysis libraries, including seaborn and scikit-learn, and set the kernel working directory to the repository root so that the four CSV paths resolve.
