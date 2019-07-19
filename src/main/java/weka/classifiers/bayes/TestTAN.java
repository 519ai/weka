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
  /** �������������Եļ������飬��ʾ����ÿ����ÿ�����������Ե�ÿ��ȡֵ�ĸ�������,
   * ���е�һά��ʾ����,�ڶ�ά��ʾ������,����ά��ʾ����ֵ,
   * ����m_Counts[yes][outlook][sunny] */
  protected double [][][] m_Counts;
  
  /** The means for numeric attributes. */
  /**��ֵ�����Եľ�ֵ����,���е�һά��ʾ����, �ڶ�ά��ʾ������,
   * ����m_Means[no][temperature], 
   * ��ʽΪ��Xmean = (X1+X2+...+Xn)/n
   * */
  protected double [][] m_Means;
  

  /** The standard deviations for numeric attributes.  ��ֵ���Եı�׼�� 
   * *��ֵ�����Եı�׼�����飬���е�һά��ʾ�������ڶ�ά��ʾ������������m_Devs[no][temperature]����ʽΪ....
   * */
  protected double [][] m_Devs;

  /** The prior probabilities of the classes.
   *  ÿ������������,��һά��ʾ����,����m_Prior[yes] 
   * */
  protected double [] m_Priors;

  /** The instances used for training.
   * ��������ѵ����ʵ��
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
  protected static double NORM_CONST = Math.sqrt(2 * Math.PI);   //��̬�ֲ����� 

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
  * ���������
  *
  * ����instances��ʾѵ��������
  * ������������������,������쳣��ʾ
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
    
    // Reserve space   ��ʼ��
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
    
    // Compute counts and sums       ���������
    Enumeration enumInsts = instances.enumerateInstances();
    while (enumInsts.hasMoreElements()) {
      Instance instance = (Instance) enumInsts.nextElement();
      if (!instance.classIsMissing()) {
	Enumeration enumAtts = instances.enumerateAttributes();
	attIndex = 0;
	while (enumAtts.hasMoreElements()) {
	  Attribute attribute = (Attribute) enumAtts.nextElement();
	  if (!instance.isMissing(attribute)) {
	    if (attribute.isNominal()) {  //������Ϊ����������,����
	      m_Counts[(int)instance.classValue()][attIndex]
		[(int)instance.value(attribute)]++;
	    } else {
	      m_Means[(int)instance.classValue()][attIndex] +=
		instance.value(attribute);
	      m_Counts[(int)instance.classValue()][attIndex][0]++;
	      //��ͣ���������ƽ����
	    }
	  }
	  attIndex++;
	}
	m_Priors[(int)instance.classValue()]++; //��������� yes/no ����++
      }
    }
    
    // Compute means    ����ƽ��ֵ
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
    
    // Compute standard deviations        �����׼��
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
    
    // Normalize counts      �������������� ��������  P(sunny|yes)
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
    
    // Normalize priors   ��������������  P(yes),P(No)
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
