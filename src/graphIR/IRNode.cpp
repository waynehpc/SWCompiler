/*
 * IRNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */


#include "IRNode.h"


template<typename Dtype>
IRNode<Dtype>::IRNode()
{
  _fatherNode = NULL;
  _childNode = NULL;
}


template<typename Dtype>
IRNode<Dtype>::~IRNode() {}
  
