/*
 *  This program is free software; you can redistribute it and/or 
 *  modify it under the terms of the GNU General Public License as 
 *  published by the Free Software Foundation; either version 2 of 
 *  the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *  TAN2B.java
 *    
 *  Author: Zhihai Wang (zhhwang@bjtu.edu.cn)
 *  Version: 2.2.0
 *  Copyright (C) 12 August 2011, Zhihai Wang:
 *  Copyright (C) 2 May 2017 Zhihai Wang
 *  Copyright (C) 7 January 2019 Zhihai Wang
 *    
 *  NB. This is the TAN of Friedman, Geiger, and Goldszmidt.
 *      "Bayesian Network Classifiers",
 *      Machine Learning, Vol. 29, 131-163, 1997.
 */

package weka.classifiers.bayes;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

//import weka.core.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

//
import weka.core.Option;
import weka.core.OptionHandler;

//
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;


//import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;

//Its parent class & interface
import weka.classifiers.AbstractClassifier;
//import weka.classifiers.Classifier;

/**
 <!-- globalinfo-start -->
 * This class for building and using a Tree Augmented Naive Bayes (TAN)
 * classifier. This method outperforms naive Bayes, yet at the same 
 * time maintains the computational simplicity (no search involved) and
 * robustness that characterizes naive Bayes.<br/> 
 * <br/>
 * For more information, see<br/>
 * 
 * <br/>
 * Friedman, N., and Goldszmidt, M. Building Classifiers using 
 * Bayesian Networks. In: Proceedings of the Thirteenth National 
 * Conference on Artificial Intelligence (AAAI 1996), 1996. pp. 1277
 * -1284.<br/>
 * 
 * <br/>
 * Friedman, N., Geiger, D., and Goldszmidt, M. Bayesian network 
 * classifiers. Machine Learning, Volume 29, Number 2-3, 1997. 
 * pp. 131-163. <br/>
 * <p/>
 <!-- globalinfo-end -->
 *
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Friedman1996,
 *    author = {Friedman, N. and Goldszmidt, M.},
 *    booktitle = {Proceedings of the Thirteenth National Conference on 
 *                 Artificial Intelligence (AAAI 1996)},
 *    pages = {1277-1284},
 *    series = {AAAI 1996},
 *    title = {Building Classifiers using Bayesian Networks},
 *    publisher = {Menlo Park, CA: AAAI Press},
 *    year = {1996}
 * }
 * </pre>
 * 
 * BibTeX:
 * <pre>
 * &#64;article{Friedman1997,
 *    author = {Friedman, N., Geiger, D., and Goldszmidt, M.},
 *    journal = {Machine Learning},
 *    number = {2-3},
 *    pages = {131-163},
 *    title = {Bayesian network classifiers},
 *    volume = {29},
 *    year = {1997}
 * }    
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -E num 
 * The estimation strategies of probabilities. Valid values are: 
 * 0 For conditional probabilities, using M-estimation plus LaPlace estimation,
 *   otherwise only using LaPlace estimation. 
 * 1 For any probability, using Laplace estimation. 
 * 2 For conditional probabilities, only using M-estimation,
 *   otherwise only using LaPlace estimation. 
 * 3 If any probability nearly equals 0, using the constant EPSLON instead. 
 * (default: 0).</pre>
 *  
 * <pre> -M 
 * If set, delete all the instances with any missing value 
 * (default: false).</pre>
 * 
 * <pre> -R 
 * Choose the root node for building the maximum spanning tree 
 * (default: set by random).</pre>
 * 
 * 
 * 
 * @author Zhihai Wang (zhhwang@bjtu.edu.cn)
 * @version $Revision: 2.2.0 $
 * <br/>
 * Copyright (C) 12 August 2011, Zhihai Wang <br/>
 * Copyright (C) 2 May 2017 Zhihai Wang   <br/> 
 * Copyright (C) 7 January 2019 Zhihai Wang <br/>
 * 
 */

/*
 * 12 August 2011, Zhihai Wang: All classifiers extends Classifier.java, 
 * and Classifier.java implements the following five interfaces: 
 *   (1) Cloneable;
 *   (2) Serializable;
 *   (3) OptionHandler;
 *   (4) CapabilitiesHandler;
 *   (5) RevisionHandler.
 * 
 */
public class TAN2B extends AbstractClassifier implements 
    OptionHandler, WeightedInstancesHandler, UpdateableClassifier,
    TechnicalInformationHandler {

  /** for serialization */
  private static final long serialVersionUID = -2224762474573467803L;

  /** The copy of the training instances */
  protected Instances m_Instances;

  /** The number of training instances in the training instances. */
  private double m_NumInstances;

  /** The number of training instances with valid class value 
   * in the dataset.*/
  private double m_SumInstances = 0;

  /** The number of attributes, including the class. */
  private int m_NumAttributes;

  /** The number of class values. */
  protected int m_NumClasses;

  /** The index of the class attribute. */
  private int m_ClassIndex;

  /** The counts for each class value. */
  private double[] m_Priors;
  
  /**
   * The smoothing strategy of estimation. Valid values are: <br/>
   *   0: For any probability, using M-estimation & Laplace, otherwise 
   *      using LaPlace estimation.<br/>
   *   1: For any probability, using Laplace estimation.<br/> 
   *   2: For any probability, using M-estimation, and otherwise using 
   *      LaPlace estimation.<br/>
   *   3: If any probability nearly equals 0, using the constant EPSLON 
   *      instead. (If any prior probability nearly equals 0, then throws
   *      an Exception.) <br/><br/> 
   *   default:  0).
   */
	private int m_Estimation = 0;
	
	/** If set, delete instances with any missing value (default: false).*/ 
	private boolean m_DelMissing = false;

  /** The number of instances with missing values, only for toString(). */
  private double m_NumOfMissings = 0;

  /** Choose the root node for building the maximum spanning tree */
  private int m_Root;
  
  /** The number of arcs in current training dataset, only for toString() */
  private int m_Arcs = 0;  
  
  /** Set debug */
  protected boolean m_debug = false;

  /**
   * The sum of attribute-class counts. m_CondiPriors[c][k] is the 
   * same as m_CondiCounts[c][k][k]
   */
  // ZHW (12 August 2011):
  private long [][] m_CondiPriors;

  /** For m_NumClasses * m_TotalAttValues * m_TotalAttValues. */
  private long [][][] m_CondiCounts;

  /** The number of values for all attributes, not including class. */
  private int m_TotalAttValues;

  /** The starting index (in m_CondiCounts matrix) of each attribute. */
  private int [] m_StartAttIndex;

  /** The number of values for each attribute. */
  private int [] m_NumAttValues;
  
  /** the number of instance of each class value */
  private int[] classins;

  /**
   * The counts (frequency) of each attribute value for the dataset. 
   * Here for security, but it can be used for weighting.
   */
  private double [] m_Frequencies;

  /** Count for P(ai, aj). Used in M-estimation */
  private int [][] AandB;

  /** The Smoothing parameter for M-estimation */
  private final double SMOOTHING = 5.0;

  /** The matrix of conditional mutual information */
  private double [][] m_CondiMutualInfo;

  /** The minimum item in the matrix of conditional mutual information */
  private double EPSILON = 1.0E-4;

  /** The array to keep track of which attribute has which parent.(Tree)*/
  private int[] m_Parents;

  
  /**
   * Returns a string describing this classifier
   * 
   * @return a description of the classifier suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {

    return "The class for building and using a Tree Augmented Naive Bayes"
        + "(TAN) classifier.This method outperforms naive Bayes, yet " 
        + "at the same time maintains the computational simplicity(no search "
        + "involved) and robustness that characterize naive Bayes.\n\n"
        + "For more information, see\n\n"
        + "  [1] Friedman, N. & Goldszmidt, M. (1996). Building classifiers using "
        + "Bayesian networks. In: The Proceedings of the National Conference "
        + "on Artificial Intelligence(pp.1277-1284).Menlo Park, CA:AAAI Press."
        + "also see \n\n"
        + "  [2] Friedman, N., Geiger,D. & Goldszmidt, M. (1997). "
        + "Bayesian Network Classifiers. Machine Learning, Vol.29,pp.131-163\n\n"
        + getTechnicalInformation().toString();
  } // End of globalInfo()
  
  
  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  @Override
  public TechnicalInformation getTechnicalInformation() {
    
    TechnicalInformation  result;
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Friedman, N., Geiger,D. & "
    		                              + "Goldszmidt, M.");
    result.setValue(Field.YEAR, "1997");
    result.setValue(Field.TITLE, "Bayesian Network Classifiers");
    result.setValue(Field.JOURNAL,"Machine Learning");
    result.setValue(Field.VOLUME,"29");
    result.setValue(Field.PAGES,"131--163");
    
    return result;
  }


  
  /**
   * Returns default capabilities of the classifier.
   *
   * @return the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    // Super is used to call the parent class
    // return an Capabilities method
    Capabilities result = super.getCapabilities();
    
    // First make it not handle any type of data set
    result.disableAll();

    // Any attribute can only be NOMINAL type, 
    // its value maybe a missing value
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // A class variable can be NOMINAL, and a missing value.
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // The least number of instances that can be processed is zero
    result.setMinimumNumberInstances(0);

    return result;
  }

  
  /**
   * Generates the classifier.
   * 
   * @param instances
   *          set of instances serving as training data
   * @exception Exception
   *              if the classifier has not been generated successfully
   */
  public void buildClassifier(Instances instances) throws Exception {
        
    // Judge whether the data set can be processed
    getCapabilities().testWithFail(instances);

    // Copy of an instance set
    m_Instances = new Instances(instances);

    // Instances number
    m_NumInstances = m_Instances.numInstances(); 

    // Attribute number
    m_NumAttributes = m_Instances.numAttributes();

    // Number of classes
    m_NumClasses = m_Instances.numClasses();
    
    // Class attribute index
    m_ClassIndex = m_Instances.classIndex();
    
    // the number of attribute values except the class variable
    m_TotalAttValues = 0;
    
    // Auxiliary array initialization
    m_NumAttValues = new int[(int) m_NumAttributes];
    m_StartAttIndex = new int[(int) m_NumAttributes];
    
    // the number of instance of each class value
    m_Priors = new double [m_NumClasses];
    
    m_CondiPriors = new long[m_NumClasses][m_NumAttributes];
    
    // Check out the class attribute
    if (m_NumClasses < 2)
      throw new Exception("Class values can't less than two!");
    
    //ZHW:
    m_SumInstances = 0;
        
    // auxiliary array assignment;
    // i is the index of all the attributes of an instance.
    for (int i = 0; i < m_NumAttributes; i++) {
      if (i != m_Instances.classIndex()) {
        m_StartAttIndex[i] = m_TotalAttValues;
        m_NumAttValues[i] = m_Instances.attribute(i).numValues();
        m_TotalAttValues += m_NumAttValues[i];
      } else {
        m_NumAttValues[i] = m_NumClasses;
      }

    }

    if (m_debug) {
      System.out.println("m_TotalAttValues" + "m_TotalAttValues");
      System.out.println("m_NumClasses: " + m_NumClasses);
      System.out.println();
      System.out.println("Auxiliary array output:m_NumAttValues?");
      print1DimensionMatrix((int) m_NumAttributes, m_NumAttValues);
      System.out.println("Auxiliary array output:m_StartAttIndex");
      print1DimensionMatrix((int) m_NumAttributes, m_StartAttIndex);
    }
    
    //
    m_CondiCounts = new long[m_NumClasses][m_TotalAttValues]
    		                                  [m_TotalAttValues];
    
    // The counts (frequency) of each attribute value for the dataset
    m_Frequencies = new double[m_TotalAttValues];

    // Take account of the instances one by one
    for (int k = 0; k < m_NumInstances; k++) {

      // check whether the instance with missing value or not
      boolean missing = false;

      Instance tempInstance = (Instance) m_Instances.instance(k);

      if (tempInstance.isMissing(m_ClassIndex)) {
        // Ignore instance with missing class value
        continue;
      } else {  
        // Depend on the parameter -M
        if (m_DelMissing) {
          // Remove the instances with missing values
          int i = 0;
          while ((i < m_NumAttributes) && (!missing)) {
            if (tempInstance.isMissing(i)) {
              missing = true;
            }
            i++;
          }
        }
      }

      if (!missing) {
        addToCounts(tempInstance);
      } else {
        m_NumOfMissings++;
      }
    }

    //
    if (m_debug) {
      System.out.println("Array: m_CondiCounts[][][]:");
      threedimout(m_NumClasses, m_TotalAttValues, m_TotalAttValues,
          m_CondiCounts);

    }
        
    // Computing conditional mutual information matrix
    m_CondiMutualInfo = new double[(int) m_NumAttributes - 1]
    		                          [(int) m_NumAttributes - 1];
    m_CondiMutualInfo = computeCMInfo();
    
    if (m_debug) {
      System.out.println("Conditional mutual information matrix output");
      twodimout((int) m_NumAttributes - 1, (int) m_NumAttributes - 1, m_CondiMutualInfo);
    }

    // Generating a one-dimensional array, get the tree
    m_Parents = prim((int) m_NumAttributes - 1, m_CondiMutualInfo);
    
    if (m_debug) {
      System.out.println("Tree output:m_Parents?");
      print1DimensionMatrix((int) m_NumAttributes - 1, m_Parents);

    }

  }
  
  /**
   * Puts an instance's values into m_CondiCounts, m_CondiPriors, m_Priors,
   * m_SumInstances, m_Frequencies.
   * 
   * @param instance the instance whose values are to be put into the counts
   *          variables
   * @Exception
   */
  private void addToCounts(Instance instance) throws Exception {

    int tempClassValue = (int) instance.classValue();

    // Add to m_Priors.
    m_Priors[tempClassValue]++;

    // The number of the instances to count
    m_SumInstances++;

    // Store this instance's attribute values into an integer array,
    int[] attIndex = new int[m_NumAttributes];

    for (int i = 0; i < m_NumAttributes; i++) {
      if (instance.isMissing(i) || i == m_ClassIndex)
        attIndex[i] = -1;
      else
        attIndex[i] = m_StartAttIndex[i] + (int) instance.value(i);
    } // end of for

    for (int att1 = 0; att1 < m_NumAttributes; att1++) {
      // avoid pointless looping
      if (attIndex[att1] == -1) continue;

      m_Frequencies[attIndex[att1]]++;

      m_CondiPriors[tempClassValue][att1]++;

      for (int att2 = 0; att2 < m_NumAttributes; att2++) {
        if (attIndex[att2] != -1) {
          m_CondiCounts[tempClassValue][attIndex[att1]]
          		                         [attIndex[att2]]++;
        }
      }
    }
  } // END OF THIS METHOD.


  /**
   * Handle each instance separately
   * 
   * @param Every
   *          instance of an instances
   */
  public void processinstance(Instance ins) {
    // index of Attribute
    for (int i = 0; i < m_NumAttributes; i++) {
      // Excluded class variable
      if (i != m_ClassIndex) {
        // index of Attribute
        for (int j = 0; j < m_NumAttributes; j++) {
          // Excluded class variable
          if (j != m_ClassIndex) {
            m_CondiCounts[(int) ins.classValue()][(int) (m_StartAttIndex[i] + ins
                .value(i))][(int) (m_StartAttIndex[j] + ins.value(j))]++;

          }
        }
      }

    }
    classins[(int) ins.classValue()]++;

  }

  /**
   * Computing mutual information matrix
   */
  public double[][] computeCMInfo() {
    // mutual information matrix
    double[][] tempCondiMutualInfo = new double[m_NumAttributes]
    		                                       [m_NumAttributes];
    double tempCMI;
    
    double finalCMI = 0;
    // Attribute Ai
    for (int att1 = 0; att1 < m_NumAttributes; att1++) {
      if (att1 != m_ClassIndex) {
        // Attribute Aj
        for (int att2 = 0; att2 < m_NumAttributes; att2++) {
          if (att2 != m_ClassIndex && att1 != att2) {
            // Value of attribute Ai
            for (int iattv = m_StartAttIndex[att1]; iattv < m_StartAttIndex[att1]
                + m_NumAttValues[att1]; iattv++) {
              // Value of attribute Aj
              for (int jattv = m_StartAttIndex[att2]; jattv < m_StartAttIndex[att2]
                  + m_NumAttValues[att2]; jattv++) {
                // Class variable C
                for (int iclass = 0; iclass < m_NumClasses; iclass++) {
                  // Calculation P(Ai,Aj,C)
                  double part1 = (double) (m_CondiCounts[iclass][iattv][jattv] + 1)
                      / (m_NumInstances + m_NumAttValues[att1]
                          + m_NumAttValues[att2] + m_NumClasses);
                  // Calculation P(Ai,Aj|C)=(count(Ai,Aj,C)+1)
                  // /(count(c)+value(Ai,Aj))
                  double part2 = (double) (m_CondiCounts[iclass][iattv][jattv] + 1)
                      / (double) (classins[iclass] + m_NumAttValues[att1] + m_NumAttValues[att2]);
                  // Calculation
                  // P(Ai|C)=(count(Ai,C)+1)/(coount(C)+value(Ai))
                  double part3 = (double) (m_CondiCounts[iclass][iattv][iattv] + 1)
                      / (double) (classins[iclass] + m_NumAttValues[att1]);
                  // Calculation P(Aj|C)
                  double part4 = (double) (m_CondiCounts[iclass][jattv][jattv] + 1)
                      / (double) (classins[iclass] + m_NumAttValues[att2]);
                  tempCMI = Math.abs(Math.log(part2 / (part3 * part4)));
                  finalCMI += part1 * tempCMI;
                }
              }

            }
            tempCondiMutualInfo[att1][att2] = finalCMI;

          }

        }

      }

    }
    return tempCondiMutualInfo;

  }

  /**
   * prim
   */
  public int[] prim(int m_NumAttributes, double[][] weight) {
    // Weights, at any time to update
    double[] lowcost = new double[m_NumAttributes];
    // Representing an array of trees
    int[] tree = new int[m_NumAttributes];
    // Tag node access
    boolean[] v = new boolean[m_NumAttributes];
    // Specify the root node, initialize
    v[m_Root] = true;
    tree[m_Root] = -1;
    for (int i = 0; i < m_NumAttributes; i++) {
      if (i != m_Root) {
        lowcost[i] = weight[m_Root][i];
        tree[i] = m_Root;
        v[i] = false;
      }

    }
    for (int i = 0; i < m_NumAttributes; i++) {
      double max = -1;
      int j = 1;
      // Search for the next node
      for (int k = 0; k < m_NumAttributes; k++) {
        if ((lowcost[k] > max) && (!v[k])) {
          max = lowcost[k];
          j = k;
        }
      }
      // Tag the node that has access to
      v[j] = true;
      // Update the remaining nodes
      for (int k = 0; k < m_NumAttributes; k++) {
        if ((weight[j][k] < lowcost[k]) && !v[k]) {
          lowcost[k] = weight[j][k];
          tree[k] = j;

        }

      }

    }
    return tree;
  }

  /**
   * Calculates the class membership probabilities for the given test instance.
   * 
   * @param instance
   *          the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception
   *              if there is a problem generating the prediction
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
    // Probability distribution matrix
    double[] probs = new double[m_NumClasses];
    // Class value
    for (int iclass = 0; iclass < m_NumClasses; iclass++) {
      probs[iclass] = 1;
      // Attribute
      for (int iatt = 0; iatt < m_NumAttributes; iatt++) {
        if (iatt != instance.classIndex()) {
          // record the parent node of attribute
          int attfather = m_Parents[iatt];
          // record the start location of attribute
          int attStartIndex = m_StartAttIndex[iatt];
          if (!instance.isMissing(iatt) && iatt != m_ClassIndex) {
            if (attfather == -1 || instance.isMissing(attfather)) {
              // Calculation
              // P(Ai|C)=(count(Ai,C)+1)/(count(C)+value(Ai))
              probs[iclass] *= ((double) (m_CondiCounts[iclass]
[attStartIndex           + (int) instance.value(iatt)][attStartIndex
                  + (int) instance.value(iatt)] + 1) / (double) (classins[iclass] + m_NumAttValues[iatt]));
            } else {
              // record the index of the parent node
              int attFatherStart = m_StartAttIndex[attfather];
              // record number of attribute values
              int num = m_NumAttValues[iatt];
              // Calculation
              // P(Aj|Ai,C)=(count(Ai,C,Aj)+1)/(count(c,Aj)+value(Ai))
              probs[iclass] *= ((double) m_CondiCounts[iclass][attStartIndex
                  + (int) instance.value(iatt)][attFatherStart
                  + (int) instance.value(attfather)] + 1)
                  / ((double) (m_CondiCounts[iclass][attFatherStart
                      + (int) instance.value(attfather)][attFatherStart
                      + (int) instance.value(attfather)] + m_NumAttValues[iatt]));

            }

          }

        }
      }

    }
    //ZHW(25 April 2017):
    /*System.out.println("probs output");
    for (int i = 0; i < m_NumClasses; i++) {
      System.out.print(probs[i] + "_");
    }
    System.out.println();*/
    return probs;

  }

  // Output of one dimensional array
  public void print1DimensionMatrix(int dim, int[] line) {
    for (int i = 0; i < dim; i++) {
      System.out.print(line[i] + "-");

    }
    System.out.println();
  }

  // Output of two dimensional array
  public void twodimout(int dim1, int dim2, double[][] line) {
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        System.out.print(line[i][j] + "-");

      }
      System.out.println();
    }

  }

  // Output of three-dimensional array
  public void threedimout(int dim1, int dim2, int dim3, long[][][] line) {
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          System.out.print(line[i][j][k] + "-");

        }
        System.out.println();
      }
      System.out.println();
    }

  }

 
 

  /**
   * Returns an enumeration describing the available options
   * 
   * @return an enumeration of all the available options
   * 
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(3);

    newVector.addElement(new Option("\tChoose Estimation Method\n", "E", 1,
                                    "-E"));

    newVector.addElement(new Option(
        "\tChoose the root node for build  the maximum " + "spanning tree\n",
        "R", 1, "-R"));
    
    newVector.addElement(new Option("\tDelete Instances having missing "
        + "values\n", "M", 0, "-M"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    
    return newVector.elements();
  }//End of 



	/**
	 * Parses a given list of options. Valid options are:
	 * <p>
	 * 
	 * -E The strategy of estimation <br>
	 * 0 For any probability, using M-estimation & Laplace estimation. For other,
	 * using Laplace estimation. 1 All Probabilities are estimated by Laplace
	 * estimation. 2 For any probability with conditional, using M-estimation. For
	 * other, using Laplace estimation. 3 If any probability nearly equals 0,using
	 * a constant EPSLON instead. (default:0)
	 * 
	 * -R <br>
	 * If true, delete the instances that include missing values(default = no).
	 * <p>
	 * 
	 * @param options the list of options as an array of strings
	 * @exception Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		
		// ZHW(25 April 2017):
		m_DelMissing = Utils.getFlag('M', options);
		
		String esti = Utils.getOption('E', options);
		if (esti.length() != 0) {
			m_Estimation = Integer.parseInt(esti);
		}

		String root = Utils.getOption('R', options);
		if (root.length() != 0) {
			m_Root = Integer.parseInt(root);

		}

		super.setOptions(options);

		// Utils.checkForRemainingOptions(options);
	} // End of setOptions()

	
	/**
	 * Gets the current settings of the classifier.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 * 
	 */
	public String[] getOptions() {

		String[] superOptions = super.getOptions();
		String[] options = new String[superOptions.length + 5];
		int current = 0;

		options[current++] = "-E";
		options[current++] = "" + m_Estimation;

		options[current++] = "-R";
		options[current++] = "" + m_Root;
		
		if (m_DelMissing) {
			options[current++] = "-M";
		}
		
		System.arraycopy(superOptions, 0, options, current, superOptions.length);
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	} // End of 

	
	/**
   * Returns a description of the classifier.
   * 
   * @return a description of the classifier as a string.
   * 返回对本分类器的描述
   */
  public String toString() {

    StringBuffer text = new StringBuffer();

    text.append("        ---- ");
    text.append("Tree-Augmented Naive Bayes Classifier ----\n");
    if (m_Instances == null) {
      text.append(": No model built yet.");
    } else {
      try {

        if (m_DelMissing) {
          text.append("\n -M: Delete any instance with missing values.\n");
        }

        switch (m_Estimation) {
        case 0:
          text.append
              ("\n E=0: For any probability, using M-estimation & Laplace.\n");
          break;
        case 1:
          text.append("\n E=1: All Probabilities are estimated by the Laplace "
                      + "estimation.\n");
          break;
        case 2:
          text.append("\n E=2: For any probability, using M-estimation.\n");
          break;
        case 3:
          text.append("\n E=3: If any probability nearly equals 0,"
                      + "using a constant EPSLON instead.\n ");
          break;
        default:
          throw new Exception("Unknown estimate type!");
        }

        // Print to string the instances
        text.append("\nThe Instances: \n");
        text.append("  The number of instances with missing values :       "
            + (int) m_NumOfMissings + ".\n");
        text.append("  The number of instances without class values :      "
            + +(int) m_SumInstances + ".\n");
        text.append("  The number of instances in the training instances:  "
            + (int) m_NumInstances + ".\n");

        // Print to string the attribute relationships
        text.append("\n\nAttributes:  \n");
        text.append(" The number of attributes without class:  "
            + (m_NumAttributes - 1) + ".\n");
        if (m_Debug)
          text.append(" The number of m_Arcs:     " + m_Arcs + ".\n");
        text.append("****************************************************\n");
        for (int i = 0; i < m_NumAttributes; i++) {
          if (m_Parents[i] != -1) {
            text.append(i + ": " + m_Instances.attribute(i).name() 
                               + "'s parent is " + m_Parents[i] + "\n");
          } else {
            text.append("\n"+ i + ": " + m_Instances.attribute(i).name() 
                + " has no parent\n\n");
          }
        }
        text.append("****************************************************\n");

      } catch (Exception ex) {
        text.append(ex.getMessage());
      }
    }
    return text.toString();
  } // End of this method.

  
  /**
   * Print the matrix of Conditional Mutual Information
   * 
   * @throws Exception if the CondiMutualInfo value is more than two
   * 在debug模式下输出条件互信息矩阵
   */
  private void printMatrix(double[][] matrix) throws Exception {

    System.out.println("The matrix is below:");

    // Delete all negative entries.
    // 删除不正确的输入
    for (int i = 0; i < matrix.length; i++) {
      System.out.print("Line " + i + ": ");

      for (int j = 0; j < i; j++) {
        // for (int j = 0; j < matrix.length; j++) {

        // Throw over bad entries.
        // if (matrix[i][j] < 0) matrix[i][j] = 0;

        System.out.print(Utils.doubleToString(matrix[i][j], 4) + "  ");

        // Some entries could be more than 1 for using m-estimation
        // 使用M-estimation时可能出现的错误的输�?
        if (matrix[i][j] > 2) {
          throw new Exception("Bad entry!");
        }
      }
      System.out.println();
    }
  }


  public static void main(String[] argv) {
    try {
      runClassifier(new TAN2B(), argv);
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println(e.getMessage());
    }
//      TAN2A tmp = new TAN2A();
//      System.out.println(tmp.globalInfo());
      
  }

  @Override
  public void updateClassifier(Instance instance) throws Exception {
    // TODO Auto-generated method stub

  }

}
