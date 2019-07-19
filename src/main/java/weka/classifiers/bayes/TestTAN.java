/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    NaiveBayesSimple.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.bayes;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import java.util.Enumeration;



public class TestTAN extends AbstractClassifier{
  
  /** for serialization */
  static final long serialVersionUID = -1478242251770381214L;

  /** All the counts for nominal attributes. */
  /** 所有名称型属性的计数数组，表示属于每个类每个名称型属性的每个取值的个数数组,
   * 其中第一维表示类名,第二维表示属性名,第三维表示属性值,
   * 比如m_Counts[yes][outlook][sunny] */
  protected double [][][] m_Counts;
  
  /** The means for numeric attributes. */
  /**数值型属性的均值数组,其中第一维表示类名, 第二维表示属性名,
   * 比如m_Means[no][temperature], 
   * 公式为：Xmean = (X1+X2+...+Xn)/n
   * */
  protected double [][] m_Means;
  

  /** The standard deviations for numeric attributes.  数值属性的标准差 
   * *数值型属性的标准差数组，其中第一维表示类名，第二维表示属性名，比如m_Devs[no][temperature]，公式为....
   * */
  protected double [][] m_Devs;

  /** The prior probabilities of the classes.
   *  每个类的先验概率,第一维表示类名,比如m_Prior[yes] 
   * */
  protected double [] m_Priors;

  /** The instances used for training.
   * 定义用于训练的实例
   *  */
  protected Instances m_Instances;
  
  /**
   * Whether to use kernel density estimator rather than normal distribution for
   * numeric attributes
   */
  protected boolean m_UseKernelEstimator = false;

  /**
   * Whether to use discretization than normal distribution for numeric
   * attributes
   */
  protected boolean m_UseDiscretization = false;

  /** Constant for normal distribution. */
  protected static double NORM_CONST = Math.sqrt(2 * Math.PI);   //正态分布常量 

  protected boolean m_displayModelInOldFormat = false;
  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */

  /**
   * Generates the classifier.
   *
   * @param instances set of instances serving as training data 
   * @exception Exception if the classifier has not been generated successfully
   *
  * 构造分类器
  *
  * 参数instances表示训练例集合
  * 若分类器不正常构造,则出现异常提示
  */

  public void buildClassifier(Instances instances) throws Exception {

    int attIndex = 0;
    double sum;
    
    // can classifier handle the data?
    getCapabilities().testWithFail(instances);

    // remove instances with missing class
    instances = new Instances(instances);
    instances.deleteWithMissingClass();
    
    System.out.println("numClasses"+instances.numClasses());
    
    System.out.println("numAttribute"+instances.numAttributes());
    
    System.out.println("numInstance"+instances.numInstances());
    
    m_Instances = new Instances(instances, 0);
    
    // Reserve space   初始化
    m_Counts = new double[instances.numClasses()]
      [instances.numAttributes() - 1][0];
    m_Means = new double[instances.numClasses()]
      [instances.numAttributes() - 1];
    m_Devs = new double[instances.numClasses()]
      [instances.numAttributes() - 1];
    m_Priors = new double[instances.numClasses()];
    
    Enumeration enu = instances.enumerateAttributes();
    while (enu.hasMoreElements()) {
      Attribute attribute = (Attribute) enu.nextElement();
      if (attribute.isNominal()) {
	for (int j = 0; j < instances.numClasses(); j++) {
		
	  m_Counts[j][attIndex] = new double[attribute.numValues()];
	  System.out.println("attribute numValues()"+attribute.numValues());
	}
      } else {
	for (int j = 0; j < instances.numClasses(); j++) {
	  m_Counts[j][attIndex] = new double[1];
	}
      }
      attIndex++;
    }
    System.out.println("nominal"+attIndex);
    
    // Compute counts and sums       计数，求和
    Enumeration enumInsts = instances.enumerateInstances();
    while (enumInsts.hasMoreElements()) {
      Instance instance = (Instance) enumInsts.nextElement();
      if (!instance.classIsMissing()) {
	Enumeration enumAtts = instances.enumerateAttributes();
	attIndex = 0;
	while (enumAtts.hasMoreElements()) {
	  Attribute attribute = (Attribute) enumAtts.nextElement();
	  if (!instance.isMissing(attribute)) {
	    if (attribute.isNominal()) {  //属性若为名称型属性,计数
	      m_Counts[(int)instance.classValue()][attIndex]
		[(int)instance.value(attribute)]++;
	    } else {
	      m_Means[(int)instance.classValue()][attIndex] +=
		instance.value(attribute);
	      m_Counts[(int)instance.classValue()][attIndex][0]++;
	      //求和，计数，求平均数
	    }
	  }
	  attIndex++;
	}
	m_Priors[(int)instance.classValue()]++; //类先验概率 yes/no 数量++
      }
    }
    
    // Compute means    计算平均值
    Enumeration enumAtts = instances.enumerateAttributes();
    attIndex = 0;
    while (enumAtts.hasMoreElements()) {
      Attribute attribute = (Attribute) enumAtts.nextElement();
      if (attribute.isNumeric()) {
	for (int j = 0; j < instances.numClasses(); j++) {
	  if (m_Counts[j][attIndex][0] < 2) {
	    throw new Exception("attribute " + attribute.name() +
				": less than two values for class " +
				instances.classAttribute().value(j));
	  }
	  m_Means[j][attIndex] /= m_Counts[j][attIndex][0];
	}
      }
      attIndex++;
    }    
    
    // Compute standard deviations        计算标准差
    enumInsts = instances.enumerateInstances();
    while (enumInsts.hasMoreElements()) {
      Instance instance = 
	(Instance) enumInsts.nextElement();
      if (!instance.classIsMissing()) {
	enumAtts = instances.enumerateAttributes();
	attIndex = 0;
	while (enumAtts.hasMoreElements()) {
	  Attribute attribute = (Attribute) enumAtts.nextElement();
	  if (!instance.isMissing(attribute)) {
	    if (attribute.isNumeric()) {
	      m_Devs[(int)instance.classValue()][attIndex] +=
		(m_Means[(int)instance.classValue()][attIndex]-
		 instance.value(attribute))*
		(m_Means[(int)instance.classValue()][attIndex]-
		 instance.value(attribute));
	    }
	  }
	  attIndex++;
	}
      }
    }
    enumAtts = instances.enumerateAttributes();
    attIndex = 0;
    while (enumAtts.hasMoreElements()) {
      Attribute attribute = (Attribute) enumAtts.nextElement();
      if (attribute.isNumeric()) {
	for (int j = 0; j < instances.numClasses(); j++) {
	  if (m_Devs[j][attIndex] <= 0) {
	    throw new Exception("attribute " + attribute.name() +
				": standard deviation is 0 for class " +
				instances.classAttribute().value(j));
	  }
	  else {
	    m_Devs[j][attIndex] /= m_Counts[j][attIndex][0] - 1;
	    m_Devs[j][attIndex] = Math.sqrt(m_Devs[j][attIndex]);
	  }
	}
      }
      attIndex++;
    } 
    
    // Normalize counts      计算名称性属性 条件概率  P(sunny|yes)
    enumAtts = instances.enumerateAttributes();
    attIndex = 0;
    while (enumAtts.hasMoreElements()) {
      Attribute attribute = (Attribute) enumAtts.nextElement();
      if (attribute.isNominal()) {
	for (int j = 0; j < instances.numClasses(); j++) {
	  sum = Utils.sum(m_Counts[j][attIndex]);
	  for (int i = 0; i < attribute.numValues(); i++) {
	    m_Counts[j][attIndex][i] =
	      (m_Counts[j][attIndex][i] + 1) 
	      / (sum + (double)attribute.numValues());
	  }
	}
      }
      attIndex++;
    }
    
    // Normalize priors   计算类的先验概率  P(yes),P(No)
    sum = Utils.sum(m_Priors);
    for (int j = 0; j < instances.numClasses(); j++)
      m_Priors[j] = (m_Priors[j] + 1) 
	/ (sum + (double)instances.numClasses());
  }

  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception if distribution can't be computed
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
    
    double [] probs = new double[instance.numClasses()];
    int attIndex;
    
    for (int j = 0; j < instance.numClasses(); j++) {
      probs[j] = 1;
      Enumeration enumAtts = instance.enumerateAttributes();
      attIndex = 0;
      while (enumAtts.hasMoreElements()) {
	Attribute attribute = (Attribute) enumAtts.nextElement();
	if (!instance.isMissing(attribute)) {
	  if (attribute.isNominal()) {
	    probs[j] *= m_Counts[j][attIndex][(int)instance.value(attribute)];
	  } else {
	    probs[j] *= normalDens(instance.value(attribute),
				   m_Means[j][attIndex],
				   m_Devs[j][attIndex]);}
	}
	attIndex++;
      }
      probs[j] *= m_Priors[j];
    }

    // Normalize probabilities
    Utils.normalize(probs);

    return probs;
  }

  /**
   * Returns a description of the classifier.
   *
   * @return a description of the classifier as a string.
   */
  public String toString() {

    if (m_Instances == null) {
      return "Naive Bayes (simple): No model built yet.";
    }
    try {
      StringBuffer text = new StringBuffer("Naive Bayes (simple)");
      int attIndex;
      
      for (int i = 0; i < m_Instances.numClasses(); i++) {
	text.append("\n\nClass " + m_Instances.classAttribute().value(i) 
		    + ": P(C) = " 
		    + Utils.doubleToString(m_Priors[i], 10, 8)
		    + "\n\n");
	Enumeration enumAtts = m_Instances.enumerateAttributes();
	attIndex = 0;
	while (enumAtts.hasMoreElements()) {
	  Attribute attribute = (Attribute) enumAtts.nextElement();
	  text.append("Attribute " + attribute.name() + "\n");
	  if (attribute.isNominal()) {
	    for (int j = 0; j < attribute.numValues(); j++) {
	      text.append(attribute.value(j) + "\t");
	    }
	    text.append("\n");
	    for (int j = 0; j < attribute.numValues(); j++)
	      text.append(Utils.
			  doubleToString(m_Counts[i][attIndex][j], 10, 8)
			  + "\t");
	  } else {
	    text.append("Mean: " + Utils.
			doubleToString(m_Means[i][attIndex], 10, 8) + "\t");
	    text.append("Standard Deviation: " 
			+ Utils.doubleToString(m_Devs[i][attIndex], 10, 8));
	  }
	  text.append("\n\n");
	  attIndex++;
	}
      }
      
      return text.toString();
    } catch (Exception e) {
      return "Can't print Naive Bayes classifier!";
    }
  }

  /**
   * Density function of normal distribution.
   * 
   * @param x the value to get the density for
   * @param mean the mean
   * @param stdDev the standard deviation
   * @return the density
   */
  protected double normalDens(double x, double mean, double stdDev) {
    
    double diff = x - mean;
    
    return (1 / (NORM_CONST * stdDev)) 
      * Math.exp(-(diff * diff / (2 * stdDev * stdDev)));
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */


  public static void main(String [] argv) {
    runClassifier(new TestTAN(), argv);
  }




}
