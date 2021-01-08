
下载最新wiki dump
```
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

抽取
```
pip install wikiextractor
wikiextractor -b 100M --json --templates --processes 30 enwiki-latest-pages-articles.xml.bz2
```

合并
```
python merge_json.py --json_path text --output_file finish_merge.json
```

清洗
```
python cleanup_dataset.py all_wiki.json finish_clean.json
```

打乱、分割训练集/验证集/测试集
```
python split_json.py --input_files cleaned_all_wiki.json --output_dir finish_split --test_percent 0.001 0
```

根据数据量，训练卡数对训练集再次切分
```
wc -l finish_split/train.json
split -d -l EACH_FILE_LINES finish_split/train.json split_data
```

整理得到最终数据集
```
mkdir final_dataset
mv split_data* final_dataset
```
