package weka.classifiers.bayes;

import java.util.Enumeration;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
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
 * Class for building and using a Tree Augmented Naive Bayes(TAN) classifier.
 * This method outperforms naive Bayes, yet at the same time maintains the
 * computational simplicity (no search involved) and robustness that
 * characterize naive Bayes. For more information, see
 * <p>
 * Friedman,N.,Geiger,D. & Goldszmidt,M. (1997). Bayesian Network Classifiers
 * Published in Machine Learning(Vol.29,pp.131-163).
 *
 * Valid options are:
 * <p>
 *
 * -E num <br>
 * The estimation strategies of probabilities. Valid values are: 0 For
 * conditional probabilities, using M-estimation plus LaPlace estimation,
 * otherwise only using LaPlace estimation. 1 For any probability, using Laplace
 * estimation. 2 For conditional probabilities, only using M-estimation,
 * otherwise only using LaPlace estimation. 3 If any probability nearly equals
 * 0, using the constant EPSLON instead. (default: 0).
 * <p>
 *
 * -M <br>
 * If set, delete all the instances with any missing value (default: false).
 * <p>
 *
 * -R <br>
 * Choose the root node for building the maximum spanning tree (default: set by
 * random).
 * <p>
 *
 * @author Zhihai Wang (zhhwang@bjtu.edu.cn)
 * @version $Revision: 3.1.0 $
 */

/*
-F "weka.filters.supervised.attribute.Discretize -R first-last
-precision 6"

-W "weka.classifiers.bayes.TAN"
-t "/Users/lance/Public/wekadata/weather.numeric.arff"
-- -output-debug-info -R 1
 */
public class TAN extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler,UpdateableClassifier,
		TechnicalInformationHandler  {

	/**
	 * The copy of the training instances. 训练集实例的复制
	 */
	protected Instances m_Instances;

	/**
	 * The number of instances in the training instances. 训练集实例数量
	 */
	private double m_NumInstances;

	/** The number of trainings with valid class values observed in dataset. */
	private double m_SumInstances = 0;

	/**
	 * The number of attributes, including the class. 属性数量（包括类）
	 */
	private int m_NumAttributes;

	/**
	 * The number of class values. 类的数量
	 */
	protected int m_NumClasses;

	/**
	 * The index of the class attribute. 类属性的索引
	 */
	private int m_ClassIndex;

	/**
	 * The counts for each class value. 每个类的频次计数
	 */
	private double[] m_Priors;

	private long[] m_CountAttribute;

	boolean m_Debug = false;

	/**
	 * The sums of attribute-class counts. 统计 属性取值,类 同时发生 频次 m_CondiPriors[c][k] is
	 * the same as m_CondiCounts[c][k][k]
	 */
	private long[][] m_CondiPriors;

	/**
	 * For m_NumClasses * m_TotalAttValues * m_TotalAttValues. 统计 类，属性值，属性值 同时发生 频次
	 */
	private long[][][] m_CondiCounts;

	/**
	 * The number of values for all attributes, not including class. 不包括类在内的所有属性的取值
	 * 总个数
	 */
	private int m_TotalAttValues;

	/**
	 * The starting index (in m_CondiCounts matrix) of each attribute. 记录每个属性的取值
	 * 的起始索引
	 */
	private int[] m_StartAttIndex;

	/**
	 * The number of values for each attribute. 每个属性取值数量
	 */
	private int[] m_NumAttValues;

	/**
	 * The counts (frequency) of each attribute value for the dataset. Here for
	 * security, but it can be used for weighting.
	 */
	// use AandB[ai][ai] to replace m_Frequencies[ai] 14th.August 2008
	// private double[] m_Frequencies;

	/**
	 * Count for P(ai, aj). 统计 ai,aj同时发生的频次
	 */
	private int[][] AandB;

	/**
	 * The Smoothing parameter for M-estimation M-estimation的平滑参数
	 */
	private final double SMOOTHING = 5.0;

	/**
	 * The matrix of conditional mutual information 条件互信息矩阵
	 */
	private double[][] m_CondiMutualInfo;

	/**
	 * The minimum item in the matrix of conditional mutual information 条件互信息矩阵最小项
	 */
	private double EPSILON = 1.0E-4;

	/**
	 * The array to keep track of which attribute has which parent. (Tree) 标示 属性的父属性
	 * 的数组。
	 */
	private int[] m_Parents;

	/**
	 * The smoothing strategy of estimation. Valid values are: 0 For any
	 * probability, using M-estimation & Laplace, otherwise using LaPlace
	 * estimation. 1 For any probability, using Laplace estimation. 2 For any
	 * probability, using M-estimation, and otherwise using LaPlace estimation.. 3
	 * If any probability nearly equals 0, using the constant EPSLON instead. (If
	 * any prior probability nearly equals 0, then throws an Exception.) (default:
	 * 0).
	 */
	private int m_Estimation = -1;

	/** If set, delete all the instances with any missing value (default: false). */
	private boolean m_DelMissing = true;

	/**
	 * Choose the root node for building the maximum spanning tree
	 * 选择用于构建最大生成树的根节点，默认-1 (default: m_Root = -1, i.e., set by random).
	 */
	private int m_Root = 0;

	/**
	 * The number of arcs in current traning dataset, only for toString()
	 * 当前训练数据集中的弧数，仅用于toString()
	 */
	private int m_Arcs = 0;

	/**
	 * The number of instances with missing values, only for toString().
	 * 缺少值的实例数，仅用于toString()。
	 */
	private double m_NumOfMissings = 0;

	/**
	 * Returns a string describing this classifier
	 *
	 * @return a description of the classifier suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for building and using a Tree Augmented Naive Bayes(TAN) "
				+ "classifier.This method outperforms naive Bayes, yet at the same "
				+ "time maintains the computational simplicity(no search involved) "
				+ "and robustness that characterize naive Bayes.\n\n" + "For more information, see\n\n"
				+ "Friedman, N. & Goldszmidt, M. (1996). Building classifiers using "
				+ "Bayesian networks. In: The Proceedings of the National Conference "
				+ "on Artificial Intelligence(pp.1277-1284).Menlo Park, CA:AAAI Press."
				+ "also see \n\n  Friedman, N., Geiger,D. & Goldszmidt, M. (1997). "
				+ "Bayesian Network Classifiers. Machine Learning, Vol.29,pp.131-163";
	} // End of globalInfo()

	private int nodenum; // 节点数量

	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {

		int attIndex = 0;
		double sum;
		// For Debugging.
		if (m_Debug) {
			System.out.println("========== Starting buildClassifier() ==========");
		}

		/* Checking */
		if (instances.checkForStringAttributes()) {
			throw new UnsupportedAttributeTypeException("TAN can't handle string attributes!");
		}

		if (instances.classAttribute().isNumeric()) {
			throw new UnsupportedClassTypeException("TAN: Class is numeric!");
		}

		// The number of the class values.
		m_NumClasses = instances.numClasses();

		if (m_NumClasses < 2) {
			throw new Exception("The dataset has no class attribute!");
		}

		// The index of the class attribute.
		m_ClassIndex = instances.classIndex();

		// The number of attributes in the dataset, including the class.
		m_NumAttributes = instances.numAttributes();

		// All attributes must be nominal!
		for (int i = 0; i < m_NumAttributes; i++) {
			Attribute attribute = (Attribute) instances.attribute(i);
			if (!attribute.isNominal()) {
				throw new Exception(
						"Attributes must be nominal. Discretize " + "the dataset with FilteredClassifer please.");
			}
		}

		// Copy the instances 实例复制
		m_Instances = new Instances(instances, 0);

		// The number of instances in the training instances. 训练集实例数量
		m_NumInstances = m_Instances.numInstances();

		// Initialize the following counters
		m_SumInstances = 0;

		// ZHW: for debugging
		// The number of values for all attributes, not including class. 除了类之外的所有属性的 取值
		// 的总数量
		m_TotalAttValues = 0;

		// Allocate space for attribute reference arrays 初始化 m[i] 存储 对应 第i+1 个属性
		// 取值的起始下标, i=0
		m_StartAttIndex = new int[m_NumAttributes];

		// The number of values for each attribute. 每个属性的取值 数量
		m_NumAttValues = new int[m_NumAttributes];

		// Allocate space for conditional mutual information, including class.
		// 给条件互信息矩阵分配空间，包括类
		m_CondiMutualInfo = new double[m_NumAttributes][m_NumAttributes];

		// 类的取值数
		m_NumClasses = instances.numClasses();

		// 先验概率
		m_Priors = new double[instances.numClasses()];

		m_Parents = new int[m_NumAttributes];

		nodenum = (m_NumAttributes - 1);

		// 初始化属性个数及每个属性的取值起始索引

		// 初始化赋值 m_StartAttIndex[attIndex]，m_NumAttValues[attIndex]，m_TotalAttValues
		int m_ind = 0;// 第一个属性第一个取值索引设为0
		Enumeration enu = instances.enumerateAttributes();
		while (enu.hasMoreElements()) {
			m_StartAttIndex[attIndex] = m_ind;
			Attribute attribute = (Attribute) enu.nextElement();
			m_NumAttValues[attIndex] = attribute.numValues(); // 每个属性 对应 取值的数量
			m_ind += attribute.numValues(); // 索引
			m_TotalAttValues += attribute.numValues(); // 全部属性取值数量

			attIndex++;
		}
		m_CountAttribute = new long[m_TotalAttValues];

		// Reserve space

		// 条件概率 统计[类][属性]
		m_CondiPriors = new long[m_NumClasses][m_TotalAttValues];

		// 条件概率 统计[类][属性][属性]
		m_CondiCounts = new long[m_NumClasses][m_TotalAttValues][m_TotalAttValues];

		m_NumInstances = instances.numInstances();

		// System.out.println("实例数：" + instances.numInstances());

		// Compute counts count(sunny,yes)， m_Priors
		Enumeration enumInsts = instances.enumerateInstances();
		while (enumInsts.hasMoreElements()) {
			Instance instance = (Instance) enumInsts.nextElement();
			if (!instance.classIsMissing()) {
				enu = instances.enumerateAttributes();
				attIndex = 0;
				while (enu.hasMoreElements()) {
					Attribute attribute = (Attribute) enu.nextElement();
					if (!instance.isMissing(attribute)) {
						if (attribute.isNominal()) { // 属性若为名称型属性,计数
							m_CountAttribute[m_StartAttIndex[attIndex] + (int) instance.value(attribute)]++;
							int attrInd = m_StartAttIndex[attIndex] + (int) instance.value(attribute); // 计算属性取值在所有取值中的索引
							m_CondiPriors[(int) instance.classValue()][attrInd]++;
							// System.out.println(" ["+(int)instance.classValue()+","+attrInd+"] "+
							// m_CondiPriors[(int)instance.classValue()][attrInd]);
						}
						// 删除有缺失值
					} else {
						instances.deleteWithMissing(attIndex);
						;
						m_NumOfMissings++;
					}
					attIndex++;
				}
				m_Priors[(int) instance.classValue()]++; // 类先验概率 yes/no 数量++
			}
		}

		int ind1 = 0, ind2 = 0;
		for (int ins = 0; ins < instances.numInstances(); ins++) {// 遍历实例
			for (int attr1 = 0; attr1 < m_NumAttributes - 2; attr1++) {// 遍历属性1
				for (int attr2 = attr1 + 1; attr2 < m_NumAttributes - 1; attr2++) {// 遍历属性2,总是在属性1后面
					Instance tempInstance = instances.get(ins);
					Attribute tempAttribute1 = tempInstance.attribute(attr1);
					Attribute tempAttribute2 = tempInstance.attribute(attr2);
					ind1 = m_StartAttIndex[attr1] + (int) tempInstance.value(tempAttribute1);
					ind2 = m_StartAttIndex[attr2] + (int) tempInstance.value(tempAttribute2);
					m_CondiCounts[(int) tempInstance.classValue()][ind1][ind2]++;
					m_CondiCounts[(int) tempInstance.classValue()][ind2][ind1]++;
					// System.out.println(" ["+(int)tempInstance.classValue()+"]["+ind1+"]["+ind2+"]
					// "+m_CondiCounts[(int)tempInstance.classValue()][ind1][ind2]);
				}
			}
		}

		// for (int i = 0; i < m_NumClasses; i++) {
		// for (int j = 0; j < m_TotalAttValues; j++) {
		// for (int k = 0; k < m_TotalAttValues; k++) {
		// System.out.print(m_CondiCounts[i][j][k] + " ");
		// }
		// System.out.println();
		// }
		// System.out.println();
		// System.out.println();
		// }

		// 计算互信息 指定类C 取 index 为 0的值
		for (int c_ValIndex = 0; c_ValIndex < instances.numClasses(); c_ValIndex++) {
			for (int i = 0; i < m_NumAttributes; i++) {
				for (int j = i + 1; j < m_NumAttributes; j++) {
					for (int val1 = 0; val1 < m_NumAttValues[i]; val1++) {
						for (int val2 = 0; val2 < m_NumAttValues[j]; val2++) {
							ind1 = m_StartAttIndex[i] + val1;
							ind2 = m_StartAttIndex[j] + val2;
							double proij_c = (m_CondiCounts[c_ValIndex][ind1][ind2] + 1.0)
									/ (m_Priors[c_ValIndex] + m_NumAttValues[i] * m_NumAttValues[j]);
							double proi_c = (m_CondiPriors[c_ValIndex][ind1] + 1.0)
									/ (m_Priors[c_ValIndex] + m_NumAttValues[i]);
							double proj_c = (m_CondiPriors[c_ValIndex][ind2] + 1.0)
									/ (m_Priors[c_ValIndex] + m_NumAttValues[j]);
							double proc = (m_Priors[c_ValIndex] + 1) / (instances.numInstances() + m_NumClasses);

							// 修改
							double proijc = (m_CondiCounts[c_ValIndex][ind1][ind2] + 1.0) / (instances.numInstances()
									+ (m_NumClasses * m_NumAttValues[i] * m_NumAttValues[j]));
							m_CondiMutualInfo[i][j] += proijc * Math.log(proij_c / (proi_c * proj_c));

						}
					}
				}
			}
		}

		for (int i = 0; i < m_NumAttributes; i++) {
			for (int j = 0; j < m_NumAttributes; j++) {
				if (i == j) {
					m_CondiMutualInfo[i][j] = 0;
				} else if (i > j) {
					m_CondiMutualInfo[i][j] = m_CondiMutualInfo[j][i];
				}
			}
		}

		// 互信息矩阵包括了类
		// 构建无向完全图，顶点集为 属性集合 undir_CompGraph[m_NumAttributes-1][m_NumAttributes-1]

		// for (int i = 0; i < m_NumAttributes; i++) {
		// for (int j = 0; j < m_NumAttributes; j++) {
		//
		// System.out.printf("%f\t", m_CondiMutualInfo[i][j]);
		// }
		// System.out.println("");
		// }

		// 3 .构造最大权生成树
		// System.out.println("执行prim算法");
		prim();

		// 最大权生成树 G 有向
		// for(int i=0;i<nodenum;i++) {
		// for(int j=0;j<nodenum;j++) {
		// System.out.print(G[i][j]+" ");
		// }
		// System.out.println();
		// }

		// for (int i = 0; i < nodenum; i++) {
		// System.out.println("i:" + m_Parents[i] + " -> " + i);
		// }

		// Normalize priors 计算类的先验概率 P(yes),P(No)

		// for (int j = 0; j < instances.numClasses(); j++)
		// m_Priors[j] = (m_Priors[j] + 1.0) / (instances.numInstances() +
		// (double)instances.numClasses());
		//

	}


	private void prim() {// 默认第一个属性先进最大生成树集合，第一个属性为根
		int setFlag[] = new int[m_NumAttributes - 1];// 顶点是否进入最大生成树集合标志
		int setN = 1;
		setFlag[m_Root] = 1;// 根据m_Root来设置根节点
		m_Parents[m_Root] = -1;// 根据m_Root来设置根节点

		int inlabel = 0;// 依赖最大时已经进集合的属性下标，父节点下标
		int outlabel = 1;// 依赖最大时未进集合的属性下标
		while (setN < m_NumAttributes) {
			double max = -100.0;
			for (int outId = 0; outId < m_NumAttributes - 1; outId++) {// 没有在最大生成树集合中的点
				if (setFlag[outId] == 1)
					continue;
				for (int inId = 0; inId < m_NumAttributes - 1; inId++) {// 已经在最大生成树集合中的点
					if (setFlag[inId] == 0)
						continue;
					if (max < m_CondiMutualInfo[outId][inId]) {
						max = m_CondiMutualInfo[outId][inId];
						inlabel = inId;
						outlabel = outId;
					}
				}
			}
			setFlag[outlabel] = 1;
			setN++;
			m_Parents[outlabel] = inlabel;
		}
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 *
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;
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
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// A class variable can be NOMINAL, and a missing value.
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// The least number of instances that can be processed is zero
		result.setMinimumNumberInstances(0);

		return result;
	}



	// laplace
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] probs = new double[instance.numClasses()];
		double alpha;
		for (int c = 0; c < m_NumClasses; c++) {
			probs[c] = 1;
			for (int attr1 = 0; attr1 < m_NumAttributes - 1; attr1++) {
				Attribute attribute1 = instance.attribute(attr1);
				int ind1 = m_StartAttIndex[attr1] + (int) instance.value(attribute1);
				if (m_Parents[attr1] != -1) {
					int attr2 = m_Parents[attr1];
					Attribute attribute2 = instance.attribute(attr2);
					int ind2 = m_StartAttIndex[attr2] + (int) instance.value(attribute2);
					double pParent_Class = (m_CondiPriors[c][ind2] + 1) / (m_Priors[c] + m_NumAttValues[attr2]);
					// double pd = (m_CondiPriors[c][ind2]+1)/(m_P)
					probs[c] *= ((double) m_CondiCounts[c][ind2][ind1] + 1)
							/ (m_CondiPriors[c][ind2] + m_NumAttValues[attr1]);
				} else {
					probs[c] *= ((double) m_CondiPriors[c][ind1] + 1) / (m_Priors[c] + m_NumAttValues[attr1]);
				}
			}
			m_Priors[c] = (m_Priors[c] + 1.0) / (m_NumInstances + (double) instance.numClasses());
			probs[c] *= m_Priors[c];
		}
		// Normalize probabilities
		Utils.normalize(probs);

		return probs;
	}




	public String toString() {
		if (m_Instances == null) {
			return "Naive Bayes (simple): No model built yet.";
		}
		try {
			StringBuffer text = new StringBuffer("Naive Bayes (simple)");
			int attIndex;

			for (int i = 0; i < m_Instances.numClasses(); i++) {
				text.append("\n\nClass " + m_Instances.classAttribute().value(i) + ": P(C) = "
						+ Utils.doubleToString(m_Priors[i], 10, 8) + "\n\n");
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
							text.append(Utils.doubleToString(m_CondiPriors[i][m_StartAttIndex[attIndex] + j], 10, 8)
									+ "\t");
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




		m_Debug = Utils.getFlag("output-debug-info",options);


		String esti = Utils.getOption('E', options);
		if (esti.length() != 0) {
			m_Estimation = Integer.parseInt(esti);
		}



		String root = Utils.getOption('R', options);
		if (root.length() != 0) {
			m_Root = Integer.parseInt(root);

		}
		if(m_Debug){
			System.out.println("new root "+m_Root);
			System.out.println("new debug "+m_Debug);
			System.out.println("new Estimation "+ m_Estimation);
			System.out.println("new DelMissing "+ m_DelMissing);
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
		String[] options = new String[superOptions.length + 6];
		int current = 0;

		options[current++] = "-E";
		options[current++] = "" + m_Estimation;

		options[current++] = "-R";
		options[current++] = "" + m_Root;

		if (m_DelMissing) {
			options[current++] = "-M";
		}
		if(m_Debug){
			options[current++] = "-output-debug-info";

		}

		System.arraycopy(superOptions, 0, options, current, superOptions.length);
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	} // End of

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



	public static void main(String args[]) throws Exception {
//

		TAN tan = new TAN();
		runClassifier(tan,args);
	}


//	不注释掉会出问题
	@Override
	public void updateClassifier(Instance instance) throws Exception {

	}
}