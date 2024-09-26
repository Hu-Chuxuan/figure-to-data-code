# Structure

There are three files: ``Plot.py``, ``SubplotConstructor.py``, and ``ImageCleaner.py``. 

The pipeline is as followed: 

1. Call ``ImageCleaner`` to preprocess the target image. 
2. Use ``SubplotConstructor`` to estimate axes and use LLM to select the correct ones to form the ``Subplot``s in a ``Plot`` (these two classes are in ``Plot.py``). 
3. Call the corresponding method to estimate the data points in each subplot given their types (dot plot, histogram, or continuous). 
4. Use the LLM to organize the recognized data points into ``Curve``s. 
5. Output the extracted data points into a csv file. 

# TODO

- [ ] Clean image and estimate axes
- [ ] Estimate data points
    - [ ] Dot points
    - [ ] Histogram
    - [ ] Continuous curves
- [ ] Organize data points and output csv files