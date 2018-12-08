#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Instance():
    
    """
    Represents an instance object: an instance of the data to be predicted or trained on
    """
    value = None
    
    
    def __init__(self, attribute_list, class_value):
        self.attributes=attribute_list
        self.value = class_value
    
    def add_attr(self, value):
        self.attributes.append(value)
        
    def getValue(self):
        return self.class_value