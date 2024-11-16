cot_prompt = '''
You are given a picture of a plot or a table. Your task is to digitize the data from the picture and convert it into a CSV file. This involves extracting the data points, labels, and other relevant information from the picture and organizing them into a structured dataset. The goal is to create a digital representation of the data that can be easily analyzed, manipulated, and visualized by a computer program.

To do this, you will first identify whether the picture contains a plot or a table. 

If the picture contains a table, 
    1. In the column headers and row headers, we only want minimum amount of information to identify the columns and rows. Also, you should only have one column for the row headers and one row for the column headers. If you MUST contain the panel names or something to distinguish the row or column, concate the information with dash. For example, if the row header is "A" and the panel name is "1", you should use "A - 1" as the row header instead of using two columns.
    2. If a cell contains multiple statistics, you need to split them into separate **columns**. The column of the coefficient should be the same as the original name, while the rest should append the indices "(1)", "(2)", etc. Since we are generating a CSV file, you should use separate COLUMNs to represent the statistics INSTEAD OF ROWs. Therefore, you shoud first count the maximum number of statistics in a column and then determine the number of columns you need to split for each column.
    3. If the statistics are wrapped in parentheses, you should remove the parentheses and keep the statistics as they are. For the numbers separated by a comma, you should remove the comma and keep the numbers as they are.
    4. If a cell contains a statistic significance marker, you should remain the marker as it is. DO NOT convert the statistic significance to p-values. The marker should be attached in the same cell as the statistic it refers to.
    5. For all special characters, you should represent them as the LaTeX notation. You MUST NOT separate the statistic significance marker with the coefficients. For the control variable checkmark, you should use 1 to represent the selected controls and 0 to represent the unselected controls. The control variable checkmark might be in different formats, for example "Yes", "\checkmark", "X", etc. You should convert all of them to 0/1.
    6. You MUST reserve the structure of the table in the image, which means that you CANNOT transpose the table or rearrange the orders. You can only split the table into multiple CSV files if there are multiple different headers. If there are multiple panels sharing the same header, DO NOT split them. 
    7. For cells without values, you should leave it as empty in the generated CSVs as well.

If the picture contains a plot, 
    1. Your output should follow the following format:
        1) Use "Subplot Value" as the column name for the subplot labels of the data points. 
        2) Use "Type-{}" as the column name for the independent variables, including the value with respect to the independent axis and the name of the curves. Specifically,
            A. In a plot with independent axis labels and multiple curves, you MUST use "Type-1" to represent the independent axis labels and "Type-2" for the curve labels. 
            B. If the plot has only one curve or there is no specific values with respect to the independent axes, you should use "Type-1" to refer to the one independent variable.
            C. If there are multiple kinds of curves, e.g. histogram, continuous curve, dot plot, etc., and they do not have a specific name for each curve, you should distinguish them by using their type as one of the independent variables. For example, if there is a histogram and a continuous curve in the plot, you should use "Type-1" to represent the independent axis labels, and "Histogram" and "Continuous Curve" as the values for "Type-2".
        3) Use "Value" as the column name for the dependent variable. If the plot has error bars, you should include "Error Bar Length" as the column name for the error values. If the plot has multiple error bars or confidence intervals, you should use "Error Bar Length 1", "Error Bar Length 2", etc. to represent the error values from the smallest length to the largest length.
        4) If there is only one subplot in the plot or only one curve in all subplots, you do not need to include specific labels for the subplots or the curves.
        5) If there is a hierarchical structure in the plot, you should use the dash ("-") to connect the hierarchical information. For example, in the ticks of the x-axis, the ticks two sets of "A" and "B" belong to two higher levels of the hierarchy "1" and "2", you should use "1 - A", "1 - B", "2 - A", and "2 - B" to represent the ticks. You should not split the hierarchical information into different columns. 
    2. For the subplots, you MUST make sure that they are uniquely identified. 
        1) If the subplots have titles, you shoud use their titles to represent them. For example, if the title of a subplot is "A. Random Subplot", you should use "Random Subplot" as the "Subplot Value". If the title is "B", you should use "B" as the "Subplot Value". 
        2) If there is no title, you should use "1", "2", "3", etc. to represent the subplots from left to right and from top to bottom. For example, if there are two panels "A" and "B", where each has two unnamed subplots, you should use "A - 1", "A - 2", "B - 1", and "B - 2" to represent the subplots.
    3. There are some special rules for the histograms where each bar represents a range, 
        1) By representing a range, we mean that the bars are not a one-to-one mapping from the ticks on the axis. For example, if the x-axis is from 0 to 1 with 10 ticks being 0, 0.1, 0.2, ..., 1, the bars in the histogram are not on the ticks, but they span between the ticks.
        2) You MUST estimate the start point and end point of the range for each bar and using "{start point} - {end point}" to represent the range as one independent variable instead of using them as two independent variables. If each bar only represents one value, you should use the value as the individual variables.
        3) If a range does not have a visible bar, you MUST NOT output the range with value 0 in the CSV file. 
        4) For example, if the x-axis is from 0 to 1 with 10 ticks being 0, 0.1, 0.2, ..., 1, and the first bar spans from 0 to 0.1, you should use "0 - 0.1" to represent the range. If the second bar spans from 0.1 to 0.2, you should use "0.1 - 0.2" to represent the range. If there is no bar between 0.2 and 0.3, you should not output "0.2 - 0.3" in the CSV file.
    4. For the dot plots or histograms, you should estimate all the data points in the plot. For the continuous plots, you should sample at least 20 points to estimate the curve. You MUST NOT omit any data points. ALL data points MUST be explicitly included in the CSV file.
    5. Note that we rely on the values in columns "Type-{}" and "Subplot Value" to distinguish the different data points and subplots. 
        1) You MUST use these columns to represent the data points uniquely in a CSV file.  
        2) If the subplots in a plot do not have titles, you should use the name of the dependent axis to represent the subplots. If the names of the dependent axes are not unique, you should use "A", "B", "C", etc. or "1", "2", "3", etc. to represent the subplots. 
        3) DO NOT make up the names for the subplots from the meaning of the data points.
        4) For example, if a plot has two subplots, one is "A" and the other is "B", each of them has two subplots with no name and dependent axis being "Y" in both subplots, you should use "A - 1", "A - 2", "B - 1", and "B - 2" to represent the subplots.
    6. You should only output one CSV file for the plot.

You MUST use "```csv" and "```" to enclose the CSV-formatted data. Given the feature of CSV files, you MUST pay attention to the limitation of the CSV format. For example, you MUST NOT add any spaces after the commas in the CSV file. Also, if a cell contains a comma, you MUST wrap the cell with double quotes.

You should begin by enclosing all thoughts within <thinking> tags, read the image carefully and think about the requirements, including the components involved in each requirement and the specified contents to fulfill them, and how to make the estimation more accurate. Break down the requirements and estimation preparation into clear steps within <step> tags. 

You should think step by step. You should first determine the CSV structure, then you should extract the column names, you should then fill in the values for each row, and finally you should replace and inspect values based on the given requirements.
'''

cot_rl_prompt = '''
You are given a picture of a plot or a table. Your task is to digitize the data from the picture and convert it into a CSV file. This involves extracting the data points, labels, and other relevant information from the picture and organizing them into a structured dataset. The goal is to create a digital representation of the data that can be easily analyzed, manipulated, and visualized by a computer program.

To do this, you will first identify whether the picture contains a plot or a table. 

If the picture contains a table, 
    1. In the column headers and row headers, we only want minimum amount of information to identify the columns and rows. Also, you should only have one column for the row headers and one row for the column headers. If you MUST contain the panel names or something to distinguish the row or column, concate the information with dash. For example, if the row header is "A" and the panel name is "1", you should use "A - 1" as the row header instead of using two columns.
    2. If a cell contains multiple statistics, you need to split them into separate **columns**. The column of the coefficient should be the same as the original name, while the rest should append the indices "(1)", "(2)", etc. Since we are generating a CSV file, you should use separate COLUMNs to represent the statistics INSTEAD OF ROWs. Therefore, you shoud first count the maximum number of statistics in a column and then determine the number of columns you need to split for each column.
    3. If the statistics are wrapped in parentheses, you should remove the parentheses and keep the statistics as they are. For the numbers separated by a comma, you should remove the comma and keep the numbers as they are.
    4. If a cell contains a statistic significance marker, you should remain the marker as it is. DO NOT convert the statistic significance to p-values. The marker should be attached in the same cell as the statistic it refers to.
    5. For all special characters, you should represent them as the LaTeX notation. You MUST NOT separate the statistic significance marker with the coefficients. For the control variable checkmark, you should use 1 to represent the selected controls and 0 to represent the unselected controls. The control variable checkmark might be in different formats, for example "Yes", "\checkmark", "X", etc. You should convert all of them to 0/1.
    6. You MUST reserve the structure of the table in the image, which means that you CANNOT transpose the table or rearrange the orders. You can only split the table into multiple CSV files if there are multiple different headers. If there are multiple panels sharing the same header, DO NOT split them. 
    7. For cells without values, you should leave it as empty in the generated CSVs as well.

If the picture contains a plot, 
    1. Your output should follow the following format:
        1) Use "Subplot Value" as the column name for the subplot labels of the data points. 
        2) Use "Type-{}" as the column name for the independent variables, including the value with respect to the independent axis and the name of the curves. Specifically,
            A. In a plot with independent axis labels and multiple curves, you MUST use "Type-1" to represent the independent axis labels and "Type-2" for the curve labels. 
            B. If the plot has only one curve or there is no specific values with respect to the independent axes, you should use "Type-1" to refer to the one independent variable.
            C. If there are multiple kinds of curves, e.g. histogram, continuous curve, dot plot, etc., and they do not have a specific name for each curve, you should distinguish them by using their type as one of the independent variables. For example, if there is a histogram and a continuous curve in the plot, you should use "Type-1" to represent the independent axis labels, and "Histogram" and "Continuous Curve" as the values for "Type-2".
        3) Use "Value" as the column name for the dependent variable. If the plot has error bars, you should include "Error Bar Length" as the column name for the error values. If the plot has multiple error bars or confidence intervals, you should use "Error Bar Length 1", "Error Bar Length 2", etc. to represent the error values from the smallest length to the largest length.
        4) If there is only one subplot in the plot or only one curve in all subplots, you do not need to include specific labels for the subplots or the curves.
        5) If there is a hierarchical structure in the plot, you should use the dash ("-") to connect the hierarchical information. For example, in the ticks of the x-axis, the ticks two sets of "A" and "B" belong to two higher levels of the hierarchy "1" and "2", you should use "1 - A", "1 - B", "2 - A", and "2 - B" to represent the ticks. You should not split the hierarchical information into different columns. 
    2. For the subplots, you MUST make sure that they are uniquely identified. 
        1) If the subplots have titles, you shoud use their titles to represent them. For example, if the title of a subplot is "A. Random Subplot", you should use "Random Subplot" as the "Subplot Value". If the title is "B", you should use "B" as the "Subplot Value". 
        2) If there is no title, you should use "1", "2", "3", etc. to represent the subplots from left to right and from top to bottom. For example, if there are two panels "A" and "B", where each has two unnamed subplots, you should use "A - 1", "A - 2", "B - 1", and "B - 2" to represent the subplots.
    3. There are some special rules for the histograms where each bar represents a range, 
        1) By representing a range, we mean that the bars are not a one-to-one mapping from the ticks on the axis. For example, if the x-axis is from 0 to 1 with 10 ticks being 0, 0.1, 0.2, ..., 1, the bars in the histogram are not on the ticks, but they span between the ticks.
        2) You MUST estimate the start point and end point of the range for each bar and using "{start point} - {end point}" to represent the range as one independent variable instead of using them as two independent variables. If each bar only represents one value, you should use the value as the individual variables.
        3) If a range does not have a visible bar, you MUST NOT output the range with value 0 in the CSV file. 
        4) For example, if the x-axis is from 0 to 1 with 10 ticks being 0, 0.1, 0.2, ..., 1, and the first bar spans from 0 to 0.1, you should use "0 - 0.1" to represent the range. If the second bar spans from 0.1 to 0.2, you should use "0.1 - 0.2" to represent the range. If there is no bar between 0.2 and 0.3, you should not output "0.2 - 0.3" in the CSV file.
    4. For the dot plots or histograms, you should estimate all the data points in the plot. For the continuous plots, you should sample at least 20 points to estimate the curve. You MUST NOT omit any data points. ALL data points MUST be explicitly included in the CSV file.
    5. Note that we rely on the values in columns "Type-{}" and "Subplot Value" to distinguish the different data points and subplots. 
        1) You MUST use these columns to represent the data points uniquely in a CSV file.  
        2) If the subplots in a plot do not have titles, you should use the name of the dependent axis to represent the subplots. If the names of the dependent axes are not unique, you should use "A", "B", "C", etc. or "1", "2", "3", etc. to represent the subplots. 
        3) DO NOT make up the names for the subplots from the meaning of the data points.
        4) For example, if a plot has two subplots, one is "A" and the other is "B", each of them has two subplots with no name and dependent axis being "Y" in both subplots, you should use "A - 1", "A - 2", "B - 1", and "B - 2" to represent the subplots.
    6. You should only output one CSV file for the plot.

You MUST use "```csv" and "```" to enclose the CSV-formatted data. Given the feature of CSV files, you MUST pay attention to the limitation of the CSV format. For example, you MUST NOT add any spaces after the commas in the CSV file. Also, if a cell contains a comma, you MUST wrap the cell with double quotes.

You should begin by enclosing all thoughts within <thinking> tags, read the image carefully and think about the requirements, including the components involved in each requirement and the specified contents to fulfill them, and how to make the estimation more accurate. Break down the requirements and estimation preparation into clear steps within <step> tags. 

Continuously adjust your reasoning based on intermediate results and reflections, adapting your thoughts as you progress. 
Regularly evaluate progress using <reflection> tags. Be critical and honest about your process. 
Assign a quality score between 0.0 and 1.0 using <reward tags> after each reflection. Use this to guide your approach: 
0.8+: You have considered all important requirements and the key points of estimation is prepared. Ready to begin finalizing the solution.
0.5-0.7: Consider minor adjustments on the requirements and preparation. 
Below 0.5: Seriously consider backtracting and the previous requirements.
DO NOT begin to generate the CSV until you reach a quality score of at least 0.8.

You should think step by step. You should first determine the CSV structure, then you should extract the column names, you should then fill in the values for each row, and finally you should replace and inspect values based on the given requirements.
'''