---
layout: post
title:  "Ignition - Roasting eggs on Amazon AWS with Python, Spark, Pandas and Scipy: Part I (The Context)"
date:   2018-04-08
comments: true
---

Project code name "Ignition" is about scaling some [PySpark](http://spark.apache.org/) computation for timeseries processing on [Amazon AWS](https://aws.amazon.com). The target is to process 1TB of timeseries on different hardware setups and extract metrics like execution time or AWS related costs. This first part introduces the context: the data and the computation that will be the subject of this experiment. The following parts will cover the performance, scalability and cost on AWS.

Regarding the data, here is an interesting dataset: [The MIMIC-III Waveform Database Matched Subset](https://physionet.org/physiobank/database/mimic3wdb/matched/).

This dataset contains thousands of [ECG recordings](https://en.wikipedia.org/wiki/Electrocardiography) which consist in heart beat signals with a rate of 125 samples per second. As the data comes in a specific format called [WFDB](https://physionet.org/physiotools/wfdb.shtml), WFDB files were converted to [Parquet](https://parquet.apache.org/) file format.

In this first part, PySpark will be used to distribute the computation on a local Spark Standalone installation.

Now lets go through the interesting bits of the code.


_**Initialise the Spark session in order to use the Spark SQL DataFrame API.**_


```python
%%time

spark = pyspark.sql                 \
               .SparkSession        \
               .builder             \
               .appName('ignition') \
               .getOrCreate()
```

    CPU times: user 8 ms, sys: 8 ms, total: 16 ms
    Wall time: 1.78 s


_**Load the list of parquet files composing the dataset.**_


```python
sdf = spark.read \
           .csv('parquet_files.csv',
                header=True,
                inferSchema=True)
sdf.limit(10).toPandas()
```




<div>
<!-- <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style> -->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>size</th>
      <th>bucket_id</th>
      <th>subject_id</th>
      <th>record_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p003495/3916659.0000.parquet</td>
      <td>89215291</td>
      <td>p00</td>
      <td>p003495</td>
      <td>p00-p003495-3916659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p003495/3916659.0001.parquet</td>
      <td>16368810</td>
      <td>p00</td>
      <td>p003495</td>
      <td>p00-p003495-3916659</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p000020/3544749.0000.parquet</td>
      <td>68988192</td>
      <td>p00</td>
      <td>p000020</td>
      <td>p00-p000020-3544749</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p000020/3544749.0001.parquet</td>
      <td>69456632</td>
      <td>p00</td>
      <td>p000020</td>
      <td>p00-p000020-3544749</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p000020/3544749.0002.parquet</td>
      <td>70660667</td>
      <td>p00</td>
      <td>p000020</td>
      <td>p00-p000020-3544749</td>
    </tr>
    <tr>
      <th>5</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p000020/3544749.0003.parquet</td>
      <td>30201955</td>
      <td>p00</td>
      <td>p000020</td>
      <td>p00-p000020-3544749</td>
    </tr>
    <tr>
      <th>6</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p000030/3524877.0000.parquet</td>
      <td>7733673</td>
      <td>p00</td>
      <td>p000030</td>
      <td>p00-p000030-3524877</td>
    </tr>
    <tr>
      <th>7</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p000033/3713820.0000.parquet</td>
      <td>74591347</td>
      <td>p00</td>
      <td>p000033</td>
      <td>p00-p000033-3713820</td>
    </tr>
    <tr>
      <th>8</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p000033/3713820.0001.parquet</td>
      <td>75399674</td>
      <td>p00</td>
      <td>p000033</td>
      <td>p00-p000033-3713820</td>
    </tr>
    <tr>
      <th>9</th>
      <td>../../data/physionet/mimic3wdb/matched/p00/p000033/3713820.0002.parquet</td>
      <td>1863999</td>
      <td>p00</td>
      <td>p000033</td>
      <td>p00-p000033-3713820</td>
    </tr>
  </tbody>
</table>
</div>



During the conversion of WFDB files to Parquet, the signal has been splitted in dataframe of 10 millions rows which produces Parquet files averaging 70MB in size.

_**Take one row and print the Parquet file size.**_


```python
row = list(
    map(lambda row: (row.record_id,
                     row.file,
                     row.size),
        sdf.select('record_id',
                   'file',
                   'size') \
           .take(1)))[0]
record_id, parquet_file, size = row

print('Parquet files size: %d bytes' % size)
```

    Parquet files size: 89215291 bytes


_**Read the parquet file with Pandas (using [Apache Arrow](https://arrow.apache.org)).**_


```python
%%time

pdf = pd.read_parquet(parquet_file)

print('Number of rows: %d' % len(pdf))
print('Memory usage: %d bytes' % pdf.memory_usage().sum())
```

    Number of rows: 10000000
    Memory usage: 400000000 bytes
    CPU times: user 1.45 s, sys: 456 ms, total: 1.9 s
    Wall time: 1.84 s


Without Parquet data compression, the size of one block of data in memory is 400MB.


```python
pdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>record</th>
      <th>parameter</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2205-05-12 14:21:31.960</td>
      <td>3916659_0001</td>
      <td>II</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2205-05-12 14:21:32.016</td>
      <td>3916659_0001</td>
      <td>II</td>
      <td>-0.398438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2205-05-12 14:21:32.024</td>
      <td>3916659_0001</td>
      <td>II</td>
      <td>-0.453125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2205-05-12 14:21:32.032</td>
      <td>3916659_0001</td>
      <td>II</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2205-05-12 14:21:32.040</td>
      <td>3916659_0001</td>
      <td>II</td>
      <td>1.500000</td>
    </tr>
  </tbody>
</table>
</div>



(Don't be afraid by the year 2205 in the time column, it is coming as is from the original WFDB file.)

_**Pivot data to tranform the dataset from narrow format to wide format** (but because there is only one parameter "II" in this dataset, I will get only one column named "II")_


```python
%%time

pdf = pdf.pivot(index='time',
                columns='parameter',
                values='value')
pdf.sort_index(inplace=True)

print('Number of rows: %d' % len(pdf))
print('Memory usage: %d bytes' % pdf.memory_usage().sum())
```

    Number of rows: 10000000
    Memory usage: 160000000 bytes
    CPU times: user 3.51 s, sys: 896 ms, total: 4.41 s
    Wall time: 4.41 s


While removing columns "record" and "parameter" the pivot is reducing the size of the data to 160MB.


```python
pdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>parameter</th>
      <th>II</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2205-05-12 05:06:01.000</th>
      <td>-0.007812</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.008</th>
      <td>-0.007812</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.016</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.024</th>
      <td>-0.007812</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.032</th>
      <td>0.007812</td>
    </tr>
  </tbody>
</table>
</div>



_**Plot the 10 first seconds of signal** (and fortunately it looks like heart beats !)_


```python
pdf[:125*10].plot(figsize=(18, 4))
plt.show()
```


![png](/assets/img/2018-04-08-ignition-part1_23_0.png)


_**Clean and down sample the data (from 125Hz to 16Hz)**_


```python
raw_pdf = pdf

# RATE = 16

pdf = pdf.fillna(method='ffill')
pdf = pdf.resample(
    '%dus' % int(1000000 / RATE)
).mean()

print('Number of rows: %d' % len(pdf))
print('Memory usage: %d bytes' % pdf.memory_usage().sum())
pdf.head()
```

    Number of rows: 1382384
    Memory usage: 22118144 bytes





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>parameter</th>
      <th>II</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2205-05-12 05:06:01.000000</th>
      <td>0.136719</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.062500</th>
      <td>0.126953</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.125000</th>
      <td>-0.040039</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.187500</th>
      <td>-0.010742</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.250000</th>
      <td>0.008789</td>
    </tr>
  </tbody>
</table>
</div>



After down sampling, the number of rows that will be processed is reduced to around 1 million and the size of the data is down to 22MB.

_**Plot the down sampled signal**_


```python
pdf[:RATE*10].plot(figsize=(18, 4))
plt.show()
```


![png](/assets/img/2018-04-08-ignition-part1_28_0.png)


The down sampled signal shows less noise but the peak's amplitude has been affected too (around 0.30 when it was 0.60 in the raw signal).

_**Here is the computation that will be applied on the signal**_


```python
def extract_features(pdf):
    result_pdf = None
    for column in pdf.columns:

        local_mins = argrelextrema(
            pdf[column].values, np.less
        )[0]

        local_maxs = argrelextrema(
            pdf[column].values, np.greater
        )[0]

        local_minmaxs = np.sort(
            np.concatenate((local_mins, local_maxs))
        )

        if result_pdf is None:
            result_pdf = pd.DataFrame()
            result_pdf[column] = pdf[column][local_minmaxs]
        else:
            tmp_pdf = pd.DataFrame()
            tmp_pdf[column] = pdf[column][local_minmaxs]
            result_pdf = pd.merge(result_pdf,
                                  tmp_pdf,
                                  left_index=True,
                                  right_index=True,
                                  how='outer' )
    return result_pdf
```

The Scipy function argrelextrema() is used to extract relative extrema (local minimums and maximums) of the down sampled signal. I choose this function for the only reason it needs to scan the signal in full while keeping the computation light (the purpose of this whole exercise is to focus on the IO-boundness of the problem ^^).

_**Apply the computation**_


```python
%%time

result_pdf = extract_features(pdf)

print('Number of rows: %d' % len(result_pdf))
print('Memory usage: %d bytes' % result_pdf.memory_usage().sum())
```

    Number of rows: 610660
    Memory usage: 9770560 bytes
    CPU times: user 1.22 s, sys: 20 ms, total: 1.24 s
    Wall time: 1.24 s


The resulting data is now about 600000 rows and 10MB.


```python
result_pdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>II</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2205-05-12 05:06:01.125000</th>
      <td>-0.040039</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.375000</th>
      <td>0.088867</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.687500</th>
      <td>0.032227</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.812500</th>
      <td>0.074219</td>
    </tr>
    <tr>
      <th>2205-05-12 05:06:01.875000</th>
      <td>0.037109</td>
    </tr>
  </tbody>
</table>
</div>



_**Plot results on the down sampled signal**_


```python
plt.figure(figsize=(18, 4))

ax = plt.subplot('111')
pdf[:RATE*10].plot(ax=ax)
result_pdf[:pdf.index[RATE*10]].plot(
    ax=ax,
    title='downsampled',
    color='r',
    style=':',
    marker='x',
    legend=False
)

plt.show()
```


![png](/assets/img/2018-04-08-ignition-part1_38_0.png)


At first sight, the detected relative extrema (coupled with linear interpolation) look like a good signal approximation but it is not defect free:<br/>
![DEFECTS](/assets/img/2018-04-08-ignition-part1_30_0_annotated.png)

_**Plot results over the original signal**_


```python
plt.figure(figsize=(18, 4))

ax = plt.subplot('111')
raw_pdf[:125*10].plot(ax=ax)
result_pdf[:pdf.index[RATE*10]].plot(
    ax=ax,
    title='raw',
    color='r',
    style=':',
    marker='x',
    legend=False
)

plt.show()
```


![png](/assets/img/2018-04-08-ignition-part1_41_0.png)


Compared to the raw signal, the approximation looks even worse given that we miss all maximum peaks.


So here is the context. We will process heart beat signals by blocks of 10 millions samples to detect local extrema. And we will do that on 1TB of data. To give you an idea, it takes around 40 seconds to process the example of signal of this blog post on one core of an Intel Core i7-6700K CPU. Using the 8 cores of the same CPU with 32GB of memory, it takes around 14 hours to process the full dataset (which total size is just under 1.2TB).

In Part II we will see how to distribute this significant piece of processing on AWS using EC2 and S3.


[See the notebook on GitHub](https://github.com/bu2/ignition/blob/master/part1/2018-04-08-ignition-part1.ipynb)
