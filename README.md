# Structure

There are three files: ``Plot.py``, ``Subplot.py``, ``SubplotConstructor.py``, and ``ImageCleaner.py``. The ``Plot`` in ``Plot.py`` controls the entire pipeline including interacting with the LLMs. 

The pipeline is as follows: 

1. Call ``ImageCleaner`` to preprocess the target image. 
2. Use ``SubplotConstructor`` to estimate axes and use LLM to select the correct axes to form the correct subplots (``DotPlot``, ``Histogram``, or ``Continuous`` in ``Subplot.py``). 
3. Call the corresponding method to estimate the data points in each subplot given their types (dot plot, histogram, or continuous). 
4. Use the LLM to organize the recognized data points into ``Curve``s. 
5. Output the extracted data points into a CSV file. 

# TODO

- [ ] Clean image (``ImageCleaner.py: ImageCleaner``) and estimate axes (``SubplotConstructor.py: SubplotConstructor``)
- [ ] Estimate data points
    - [ ] Dot points (``Subplot.py: DotPlot``)
    - [ ] Histogram (``Subplot.py: Histogram``)
    - [ ] Continuous curves (``Subplot.py: Continuous``)
- [ ] Organize data points and output CSV files