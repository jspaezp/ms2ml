#!/bin/bash

sqlite3 pan_human_library_600to603.dlib ".headers ON" ".mode csv" "select * from entries limit 5;" ".exit" > glimpse.csv
