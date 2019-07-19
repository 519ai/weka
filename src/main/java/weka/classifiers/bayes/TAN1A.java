/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * TAN1A.java
 *
 * Author: Zhihai Wang (zhhwang@bjtu.edu.cn) Version: 1.2.0
 *
 * Copyright (C) 12 August 2011, Zhihai Wang;
 * Copyright (C) 2 May 2017, Zhihai Wang;
 * Copyright (C) 24 January 2019, Zhihai Wang
 *
 * NB. This is the simple version TAN. All probability estimations are
 *     smoothed by the Laplace estimation.
 */

//Java
package weka.classifiers.bayes;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

// Its parent class & interface
import weka.classifiers.AbstractClassifier;
// checking for the training dataset
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
// import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
// there are some parameters need to be set in this program
import weka.core.Option;
// reference for this program
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> This class for building and using a
 * simple TAN
 * classifier. This method outperforms naive Bayes, yet at the same
 * time
 * maintains the computational simplicity (no search involved) and
 * robustness
 * that characterizes naive Bayes.<br/>
 * <br/>
 * For more information, see<br/>
 * <p>
 * <br/>
 * Friedman, N., and Goldszmidt, M. Building Classifiers using
 * Bayesian
 * Networks. In: Proceedings of the Thirteenth National Conference
 * on Artificial
 * Intelligence (AAAI 1996), 1996. pp. 1277-1284.<br/>
 * <p>
 * <br/>
 * Friedman, N., Geiger, D., and Goldszmidt, M. Bayesian network
 * classifiers.
 * Machine Learning, Volume 29, Number 2-3, 1997. pp. 131-163. <br/>
 * <p/>
 * <!-- globalinfo-end -->
 * <p>
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;inproceedings{Friedman1996,
 *    author = {Friedman, N. and Goldszmidt, M.},
 *    booktitle = {Proceedings of the Thirteenth National Conference
 *                 on Artificial Intelligence (AAAI 1996)},
 *    pages = {1277-1284},
 *    series = {AAAI 1996},
 *    title = {Building Classifiers using Bayesian Networks},
 *    publisher = {Menlo Park, CA: AAAI Press},
 *    year = {1996}
 * }
 * </pre>
 * <p>
 * BibTeX:
 *
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
 * <!-- technical-bibtex-end -->
 * <p>
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 *  -M
 * If set, delete all the instances with any missing value
 * (default: false).
 * </pre>
 *
 * <pre>
 *  -R
 * Choose the root node for building the maximum spanning tree
 * (default: the first attribute).
 * </pre>
 * <p>
 * <!-- options-end -->
 *
 * @author Zhihai Wang (zhhwang@bjtu.edu.cn)
 * @version $Revision: 1.2.0 $ <br/>
 * Copyright (C) 12 August 2011, Zhihai Wang <br/>
 * Copyright (C) 2 May 2017 Zhihai Wang <br/>
 * Copyright (C) 24 January 2019 Zhihai Wang <br/>
 */
public class TAN1A extends AbstractClassifier implements TechnicalInformationHandler {

    /**
     * for serialization
     */
    private static final long serialVersionUID =
            -2224762474573467803L;

    /**
     * The copy of the training instances
     */
    protected Instances m_Instances;

    /**
     * The number of training instances in the training instances
     */
    private long m_NumInstances;

    /**
     * The number of qualified training instances with valid class
     */
    private long m_SumInstances = 0;

    /**
     * The number of attributes, including the class.
     */
    private int m_NumAttributes;

    /**
     * The number of class values.
     */
    protected int m_NumClasses;

    /**
     * The index of the class attribute, not always the last one
     */
    private int m_ClassIndex;

    /**
     * The counts for each class value.
     */
    private long[] m_Priors;

    /**
     * If set, delete instances with any missing value
     */
    private boolean m_DelMissing = false;

    /**
     * Choose the root node for building the maximum spanning tree
     */
    private int m_Root = 0;

    /**
     * The number of values of attributes, not including class.
     */
    private int m_TotalAttValues;

    /**
     * The starting index (in m_CondiCounts) of each attribute.
     */
    private int[] m_StartAttIndex;

    /**
     * The number of values for each attribute.
     */
    private int[] m_NumAttValues;

    /**
     * For m_NumClasses * m_TotalAttValues * m_TotalAttValues.
     */
    private long[][][] m_CondiCounts;

    // The number of instances with missing values,
    // only for toString().
    private double m_NumOfMissings = 0;

    // Number of arcs in current training data, only for toString()
    private int m_Arcs = 0;

    // The counts (frequency) of each attribute value for the dataset.
    // Here for security, but it can be used for weighting.
    private double[] m_Frequencies;

    /**
     * The matrix of conditional mutual information
     */
    private double[][] m_CondiMutualInfo;

    /**
     * Keep track of which attribute has which parent.(Tree)
     */
    private int[] m_Parents;

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for
     * displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {
        return "The class for building and using a Tree Augmented " +
                "Naive "
                + "Bayes (TAN) classifier.This method outperforms " +
                "naive "
                + "Bayes, yet at the same time maintains the " +
                "computational "
                + "simplicity (no search involved) and robustness " +
                "that "
                + "characterize naive Bayes.\n\n For more " +
                "information, see\n\n"
                + "  [1] Friedman, N. & Goldszmidt, M. (1996). " +
                "Building "
                + "classifiers using Bayesian networks. In: The " +
                "Proceedings "
                + "of the National Conference on Artificial " +
                "Intelligence " + "(pp.1277-1284).Menlo Park, " +
                "CA:AAAI Press."
                + "also see \n\n" + "  [2] Friedman, N., Geiger,D. " +
                "& Goldszmidt, M. (1997). "
                + "Bayesian Network Classifiers. Machine Learning, " +
                "Vol.29, " + "pp.131-163\n\n"
                + getTechnicalInformation().toString();
    } // end of globalInfo()

    /**
     * Returns an instance of a TechnicalInformation object,
     * containing detailed
     * information about the technical background of this class, e.g
     * ., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR,
                "Friedman, N., Geiger,D. & " + "Goldszmidt, M.");
        result.setValue(Field.YEAR, "1997");
        result.setValue(Field.TITLE, "Bayesian Network Classifiers");
        result.setValue(Field.JOURNAL, "Machine Learning");
        result.setValue(Field.VOLUME, "29");
        result.setValue(Field.PAGES, "131--163");

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

        // any attribute can only be NOMINAL type,
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        // its value maybe a missing value
        result.enable(Capability.MISSING_VALUES);

        // any class variable can only be NOMINAL, or missing
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // The least number of instances that can be processed is zero
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data
     * @throws Exception if a classifier has not been generated
     *                   successfully
     */
    public void buildClassifier(Instances instances) throws Exception {

        // Judge whether the data set can be processed
        getCapabilities().testWithFail(instances);

        // Copy of an instance set
        m_Instances = new Instances(instances);

        // Instances number
        m_NumInstances = m_Instances.numInstances();

        // the number of attributes, including the class
        m_NumAttributes = m_Instances.numAttributes();

        // Number of classes
        m_NumClasses = m_Instances.numClasses();

        // Class attribute index
        m_ClassIndex = m_Instances.classIndex();

        // the number of instance of each class value
        m_Priors = new long[m_NumClasses];

        // the number of attribute values except the class variable
        m_TotalAttValues = 0;

        // Auxiliary array initialization
        m_NumAttValues = new int[m_NumAttributes];
        m_StartAttIndex = new int[m_NumAttributes];

        // Check out the class attribute
        if (m_NumClasses < 2)
            throw new Exception("Class values can't less than two!");

        // auxiliary array assignment;
        // i is an attribute index of an instance.
        for (int i = 0; i < m_NumAttributes; i++) {
            if (i != m_ClassIndex) {
                m_StartAttIndex[i] = m_TotalAttValues;
                m_NumAttValues[i] =
                        m_Instances.attribute(i).numValues();
                m_TotalAttValues += m_NumAttValues[i];
            } else {
                m_StartAttIndex[i] = m_TotalAttValues;
                m_NumAttValues[i] = m_NumClasses;
            }
        }

        if (m_Debug) {
            System.out.println();
            System.out.println("m_TotalAttValues= " + m_TotalAttValues);
            System.out.println("m_NumClasses=  " + m_NumClasses);
            System.out.print("m_NumAttValues = ");
            print1D_Matrix(m_NumAttributes, m_NumAttValues);
            System.out.print("m_StartAttIndex = ");
            print1D_Matrix(m_NumAttributes, m_StartAttIndex);
        }

        // allocate space
        m_CondiCounts =
                new long[m_NumClasses][m_TotalAttValues][m_TotalAttValues];

        // The counts (frequency) of each attribute value for the
        // dataset
        m_Frequencies = new double[m_TotalAttValues];

        // Take account of the instances one by one
        for (int k = 0; k < m_NumInstances; k++) {
            // check whether the instance with missing value or not
            boolean missing = false;

            Instance tempInstance =
                    (Instance) m_Instances.instance(k);

            if (tempInstance.isMissing(m_ClassIndex)) {
                // Ignore instance with missing class value
                continue;
            } else {
                // Depend on the parameter -M
                if (m_DelMissing) {
                    // Remove the instances with a missing value
                    int i = 0;
                    while ((i < m_NumAttributes) && (!missing)) {
                        if ((tempInstance.isMissing(i)) && (i != m_ClassIndex)) {
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
        } // end of for

        // ZHW：
        if (m_Debug) {
            System.out.println("m_CondiCounts[][][]:");
            print3D_Matrix(m_NumClasses, m_TotalAttValues,
                    m_TotalAttValues, m_CondiCounts);

        }

        print3D_Matrix(m_NumClasses, m_TotalAttValues,
                m_TotalAttValues, m_CondiCounts);
        // Computing conditional mutual information matrix
        m_CondiMutualInfo =
                new double[m_NumAttributes][m_NumAttributes];
        m_CondiMutualInfo = condiMutualInfoMatrix();

        // 输出
        for (int i = 0; i < m_NumClasses; i++) {
            System.out.println(i + " " + m_Priors[i]);
        }

        for (int i = 0; i < m_NumClasses; i++) {
            for (int j = 0; j < m_TotalAttValues; j++) {
                System.out.print(m_CondiCounts[i][j][j] + "\t");
            }
            System.out.println();
        }

        for (int i = 0; i < m_NumAttributes; i++) {
            for (int j = 0; j < m_NumAttributes; j++) {
                System.out.printf("%f\t", m_CondiMutualInfo[i][j]);
            }
            System.out.println();
        }

        if (m_Debug) {
            System.out.println("Conditional Mutual Information " +
                    "Matrix: ");
            print2D_Matrix(m_NumAttributes, m_NumAttributes,
                    m_CondiMutualInfo);
        }

        // ZHW(27 January 2019): data structure text book
        /*
         * double [][] testMatrix = {{-99,-19,-99,-99,-14,-99,-18,
         * -99}, {-19,-99, -5,
         * -7,-12,-99,-99,-99}, {-99, -5,-99, -3,-99,-99,-99,-99},
         * {-99, -7, -3,-99,
         * -8,-21,-99,-99}, {-14,-12, -99, -8,-99,-99,-16,-99},
         * {-99,-99,-99,-21,-99,-99,-27,-99}, {-18,-99,-99,-99,-16,
         * -27,-99,-99},
         * {-99,-99,-99,-99,-99,-99,-99,-99}}; m_ClassIndex =7;
         * m_NumAttributes=8; int
         * testRoot = 0;
         *
         * System.out.println("testMatrix："); for(int i=0;
         * i<m_NumAttributes;i++){
         * for(int j=0;j<m_NumAttributes;j++){ System.out.printf
         * ("%f\t",
         * testMatrix[i][j]); } System.out.println(); } m_Parents =
         * MaxSpanTree(testRoot, testMatrix);
         */

        // Generating a one-dimensional array, get the tree
        // default: the root node is attribute 0
        m_Parents = MaxSpanTree((int) m_Root, m_CondiMutualInfo);

        for (int i = 0; i < m_NumAttributes; i++) {
            System.out.println("Test: m_Parents:" + i + "==>" + m_Parents[i]);
        }

        if (m_Debug) {
            System.out.println("Maximum Spanning Tree: m_Parents");
            print1D_Matrix(m_NumAttributes, m_Parents);
        }
    }// End of buildClassifier()

    /**
     * Puts an instance's values into m_CondiCounts, m_Priors,
     * m_SumInstances,
     * m_Frequencies.
     *
     * @param instance the instance whose values are to be put into
     *                 all the counts
     *                 variables
     * @Exception
     */
    private void addToCounts(Instance instance) throws Exception {

        int tempClassValue = (int) instance.classValue();

        // Add to m_Priors.
        m_Priors[tempClassValue]++;

        // The number of the instances to count
        m_SumInstances++;

        // Store this instance's attribute values into an integer
        // array,
        int[] attIndex = new int[m_NumAttributes];

        for (int i = 0; i < m_NumAttributes; i++) {
            if (instance.isMissing(i) || i == m_ClassIndex)
                attIndex[i] = -1;
            else
                attIndex[i] =
                        m_StartAttIndex[i] + (int) instance.value(i);
        } // end of for


        for (int att1 = 0; att1 < m_NumAttributes; att1++) {
            // avoid pointless looping
            if (attIndex[att1] == -1)
                continue;

            m_Frequencies[attIndex[att1]]++;

            for (int att2 = 0; att2 < m_NumAttributes; att2++) {
                if (attIndex[att2] != -1) {
                    m_CondiCounts[tempClassValue][attIndex[att1]][attIndex[att2]]++;

                }
            }
        }
    } // end of addToCounts()

    /**
     * Computing conditional mutual information matrix
     *
     * @throws Exception
     */
    public double[][] condiMutualInfoMatrix() throws Exception {
        // mutual information matrix
        double[][] tempCondiMutualInfo =
                new double[m_NumAttributes][m_NumAttributes];

        // attribute A1
        for (int att1 = 0; att1 < m_NumAttributes; att1++) {
            if (att1 != m_ClassIndex) {
                // attribute A2
                for (int att2 = 0; att2 < att1; att2++) {
                    double sum = 0;
                    if (att2 != m_ClassIndex) {
                        // values of attribute A1
                        for (int i = m_StartAttIndex[att1]; i < m_StartAttIndex[att1] + m_NumAttValues[att1]; i++) {
                            // System.out.println("indexi"+i);
                            // values of attribute A2
                            for (int j = m_StartAttIndex[att2]; j < m_StartAttIndex[att2] + m_NumAttValues[att2]; j++) {
                                // the class attribute
                                for (int c = 0; c < m_NumClasses; c++) {
                                    // calculate P(Ai,Aj,C)

                                    double part1 =
                                            (double) (m_CondiCounts[c][i][j] + 1) / (double) (m_SumInstances
                                                    + m_NumAttValues[att1] * m_NumAttValues[att2] * m_NumClasses);
                                    // System.out.println("P(i,j,c)
                                    // :"+i+" "+j+" "+c+" "+part1);
                                    // calculate
                                    // P(Ai,Aj|C)=(m_CondiCounts
                                    // (Ai,Aj,C)+1) /
                                    // (m_Priors[c] + values(Ai,Aj)
                                    double part2 =
                                            (double) (m_CondiCounts[c][i][j] + 1)
                                                    / (double) (m_Priors[c] + m_NumAttValues[att1] * m_NumAttValues[att2]);

                                    // System.out.println("P(i,j|C)
                                    // " + " " + c + " " + i + " " +
                                    // j + " " + part2);
                                    // calculate P(Ai|C) =
                                    // (m_CondiCounts(Ai,Ai,C) + 1)/
                                    // (m_Priors(C) + values(Ai))
                                    double part3 =
                                            (double) (m_CondiCounts[c][i][i] + 1)
                                                    / (double) (m_Priors[c] + m_NumAttValues[att1]);
                                    // System.out.println("P(i|C)
                                    // "+"\t\t"+c+"\t\t"+i+"\t\t
                                    // "+part3+"\t\t"+att1);

                                    // calculate P(Aj|C) =
                                    // (m_CondiCounts(Aj,Aj,C) + 1)/
                                    // (m_Priors(C) + values(Aj))
                                    double part4 =
                                            (double) (m_CondiCounts[c][j][j] + 1)
                                                    / (double) (m_Priors[c] + m_NumAttValues[att2]);
                                    // System.out.println("P(j|C)
                                    // "+part4);
                                    // System.out.println
                                    // (c+"\t\t"+j+"\t\t"+part4);
                                    //
                                    // System.out.println
                                    // ("temp="+temp);
                                    double temp =
                                            Math.log(part2 / (part3 * part4));
                                    // System.out.println
                                    // ("temp"+c+" "+temp);
                                    // ZHW(27 January 2019)

                                    // if (temp < 0) throw new
                                    // Exception("Bad log!");

                                    // temp = Math.abs(Math.log
                                    // (part2 / (part3 * part4)));
                                    sum += part1 * temp;

                                    // temp = temp + (part1 * temp);

                                    // System.out.println("sum=" +
                                    // sum);
                                    // System.out.println("temp=" +
                                    // temp);
                                }
                                // System.out.println("temp"+temp);
                            }
                        }

                    }
                    // an item in the matrix
                    tempCondiMutualInfo[att1][att2] = sum;

                    // symmetric matrix
                    tempCondiMutualInfo[att2][att1] = sum;
                } // attribute Aj
            }
        } // attribute Ai

        //
        return tempCondiMutualInfo;
    } // end of condiMutualInfoMatrix()

    /**
     * Build the maximum spanning tree,
     * <p>
     * Zhang Fan (25 Nov. 2003): using the Prim Algorithm Zhang Fan
     * (15 Sept. 2004):
     * modified Zhihai Wang (24 January 2019): rewrite
     *
     * @param start  the root node
     * @param matrix keeping the weigth between vertices
     * @return an array describing the maximum spanning tree
     * @throws Exception if the start is class attribute
     */
    private int[] MaxSpanTree(int start, double[][] matrix) throws Exception {

        // data structure of the spanning tree
        int[] tree = new int[m_NumAttributes];

        // ZHW(27 January 2019)
        if (start == m_ClassIndex)
            throw new Exception("Invild root!");

        // specify the root node
        int root = start;

        // tree[root] = -1;

        // all attributes are divided into two sets: visited and
        // unvisited
        boolean[] visited = new boolean[m_NumAttributes];
        int[] parentNode = new int[m_NumAttributes];
        visited[root] = true;

        // weights, at any time to update
        double[] maxcost = new double[m_NumAttributes];
        // for (int i = 0; i < m_NumAttributes; i++) maxcost[i] = -1;
        for (int i = 0; i < m_NumAttributes; i++)
            maxcost[i] = -99;

        // initialization
        for (int i = 0; i < m_NumAttributes; i++) {
            tree[i] = -1;
            if ((i != m_ClassIndex) && (i != root)) {
                maxcost[i] = matrix[root][i];
                visited[i] = false;
                parentNode[i] = root;
            }
        }

        //
        if (m_Debug) {
            System.out.println("maxcost: ");
            for (int i = 0; i < m_NumAttributes; i++)
                System.out.print(maxcost[i] + "  ");
            System.out.println();
        }

        // find out the maximum arc
        for (int i = 0; i < m_NumAttributes - 2; i++) {
            // double max = -1;
            double max = -99;
            int maxIndex = root;
            // Search for the next node
            for (int k = 0; k < m_NumAttributes; k++) {
                if ((k != m_ClassIndex) && (!visited[k])) {
                    if (maxcost[k] > max) {
                        max = maxcost[k];
                        maxIndex = k;
                    }
                }
            }

            // Tag the node that has access to
            tree[maxIndex] = parentNode[maxIndex];
            visited[maxIndex] = true;

            // Update the remaining nodes
            for (int k = 0; k < m_NumAttributes; k++) {
                if (k != m_ClassIndex) {
                    if ((matrix[maxIndex][k] > maxcost[k]) && !visited[k]) {
                        maxcost[k] = matrix[maxIndex][k];
                        parentNode[k] = maxIndex;
                    }
                }
            }
        }
        return tree;
    } // end of MaxSpanTree()

    /**
     * Calculates the class membership probabilities for the given
     * test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @throws Exception if there is a problem generating the
     *                   prediction
     */
    public double[] distributionForInstance(Instance instance) throws Exception {

        // probability distribution vector
        double[] probs = new double[m_NumClasses];
        // Class value
        for (int c = 0; c < m_NumClasses; c++) {
            probs[c] = 1;
            // Attribute
            for (int i = 0; i < m_NumAttributes; i++) {
                if (i != m_ClassIndex) {
                    // record the parent node of attribute
                    int parent = m_Parents[i];
                    // record the start location of attribute
                    int attStartIndex = m_StartAttIndex[i];
                    if (!instance.isMissing(i)) {
                        if (parent == -1 || instance.isMissing(parent)) {
                            // P(Ai|C)=(count(Ai,C)+1)/(count(C)
                            // +value(Ai))
                            probs[c] *= ((double) (m_CondiCounts[c][attStartIndex
                                    + (int) instance.value(i)][attStartIndex + (int) instance.value(i)] + 1)
                                    / (double) (m_Priors[c] + m_NumAttValues[i]));
                        } else {
                            // record the index of the parent node
                            int parentStart = m_StartAttIndex[parent];

                            // P(Aj|Ai,C)=(count(Ai,C,Aj)+1)/(count
                            // (c,Ai)+value(Aj))
                            System.out.println("i|parent c" + (attStartIndex + (int) instance.value(i)) + "\t"
                                    + (parentStart + (int) instance.value(parent)) + "\t" + c + " "
                                    + m_CondiCounts[c][attStartIndex + (int) instance.value(i)][parentStart
                                    + (int) instance.value(parent)]);
                            probs[c] *= ((double) m_CondiCounts[c][attStartIndex + (int) instance.value(i)][parentStart
                                    + (int) instance.value(parent)] + 1)
                                    / ((double) (m_CondiCounts[c][parentStart
                                    + (int) instance.value(parent)][parentStart + (int) instance.value(parent)]
                                    + m_NumAttValues[i]));

                        }
                    } // end of if
                }
            } // end of for

            probs[c] *= ((m_Priors[c] + 1.0) / (m_NumInstances + (double) instance.numClasses()));
        }

        // ZHW(25 April 2017):
        if (m_Debug) {
            System.out.println("probs output");
            for (int i = 0; i < m_NumClasses; i++) {
                System.out.print(probs[i] + "_");
            }
            System.out.println();
        }

        return probs;
    } // End of distributionForInstance()

    /**
     * Print out an 1-dimensional array (vector)
     *
     * @param
     * @return no
     * @throws Exception if
     */
    public void print1D_Matrix(int dim, int[] line) {
        for (int i = 0; i < dim; i++) {
            if (i == (dim - 1)) {
                System.out.print(line[i]);
            } else {
                System.out.print(line[i] + " $ ");
            }
        }
        System.out.println();
    } // end of print1D_Matrix()

    /**
     * Print the matrix of Conditional Mutual Information
     *
     * @param
     * @return no
     * @throws Exception if the CondiMutualInfo value is more than two
     */
    public void print2D_Matrix(int dim1, int dim2,
                               double[][] matrix) throws Exception {
        System.out.println("The matrix is below:");
        // Delete all negative entries.
        for (int i = 0; i < matrix.length; i++) {
            System.out.print("Line " + i + ": ");
            for (int j = 0; j < i; j++) {
                // Throw over bad entries.
                if (matrix[i][j] < 0)
                    matrix[i][j] = 0;

                System.out.print(Utils.doubleToString(matrix[i][j],
                        4) + "  ");

                // Some entries could be more than 1 for using
                // m-estimation
                if (matrix[i][j] > 2)
                    throw new Exception("Bad entry!");
            }
            System.out.println();
        }
    }// end of print2D_Matrix()

    /**
     * Print out a 3-dimensional array (vector)
     *
     * @param
     * @return no
     * @throws Exception if
     */
    public void print3D_Matrix(int dim1, int dim2, int dim3,
                               long[][][] line) {
        System.out.println("3-dimensional matrix is below:");
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    System.out.print(line[i][j][k] + "-");
                }
                System.out.println();
            }
            System.out.println();
        }
    }// end of print3D_Matrix

    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    public Enumeration<Option> listOptions() {

        // Vector newVector = new Vector(3);
        Vector<Option> newVector = new Vector<Option>(2);

        //
        newVector.addElement(
                new Option("\tChoose the root node for build  the " +
                        "maximum " + "spanning tree\n", "R", 1, "-R"
                ));

        newVector.addElement(new Option("\tDelete Instances having " +
                "missing " + "values\n", "M", 0, "-M"));

        Enumeration<Option> enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        // ZHW:
        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }// End of

    /**
     * Parses a given list of options. Valid options are:
     * <p>
     * <p>
     * -E The strategy of estimation <br>
     * <p>
     * -R <br>
     * If true, delete the instances that include missing values
     * (default = no).
     * <p>
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        // ZHW(25 April 2017):
        m_DelMissing = Utils.getFlag('M', options);

        //
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
     */
    public String[] getOptions() {

        String[] superOptions = super.getOptions();
        String[] options = new String[superOptions.length + 5];
        int current = 0;

        // options[current++] = "-E";
        // options[current++] = "" + m_Estimation;

        options[current++] = "-R";
        options[current++] = "" + m_Root;

        if (m_DelMissing) {
            options[current++] = "-M";
        }

        System.arraycopy(superOptions, 0, options, current,
                superOptions.length);
        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    } // end of

    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier as a string.
     */
    public String toString() {

        StringBuffer text = new StringBuffer();

        text.append("        ---- ");
        text.append(" Simple TAN Classifier ----\n");
        if (m_Instances == null) {
            text.append(": No model built yet.");
        } else {
            try {

                if (m_DelMissing) {
                    text.append("\n -M: Delete any instance with " + "missing values.\n");
                }

                // Print to string the instances
                text.append("\nThe Instances: \n");
                text.append(
                        "  The number of instances with missing " +
                                "values :" + "       " + (int) m_NumOfMissings + ".\n");
                text.append(
                        "  The number of instances without class " +
                                "values :" + "      " + (int) m_SumInstances + ".\n");
                text.append(
                        "  The number of instances in the training "
                                + "instances:  " + (int) m_NumInstances + ".\n");

                // Print to string the attribute relationships
                text.append("\n\nAttributes:  \n");
                text.append(" The number of attributes without " +
                        "class:  " + (m_NumAttributes - 1) + ".\n");
                if (m_Debug)
                    text.append(" The number of m_Arcs:     " + m_Arcs + ".\n");
                text.append("****************************************************\n");
                for (int i = 0; i < m_NumAttributes; i++) {
                    if (m_Parents[i] != -1) {
                        text.append(i + ": " + m_Instances.attribute(i).name() + "'s parent is " + m_Parents[i] + "\n");
                    } else {
                        text.append("\n" + i + ": " + m_Instances.attribute(i).name() + " has no parent\n\n");
                    }
                }
                text.append("****************************************************\n");

            } catch (Exception ex) {
                text.append(ex.getMessage());
            }
        }
        return text.toString();
    } // End of this method.

    public static void main(String[] argv) {

        try {
            runClassifier(new TAN1A(), argv);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println(e.getMessage());
        }
    }
} // END！
