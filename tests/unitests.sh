#!/bin/bash
echo "-----------------installing pylama and doctest------------"
#pip install pylama
echo "-----------------pylama succesfully installed--------------"
echo "running code quality review with pylama ..."
pylama --ignore=E501 ../src/python/*.py
echo "-----------------code review finished---------------------"
echo "running unittests with doctests..."
python -m doctest -v ../src/python/*.py
echo "-----------------unittests finished---------------------"
echo "removing created files..."
