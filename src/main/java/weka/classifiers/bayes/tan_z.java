package weka.classifiers.bayes;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;

/**
 * @author zyw
 * @version $Revision: 4.1.0 $
 */
public class tan_z extends AbstractClassifier {

	/** The copy of the training instances.复制实例 */
	protected Instances m_Instances;

	/** The number of instances in the training instances. 存放实例的数量 */
	private double m_NumInstances;

	/** The number of trainings with valid class values observed in dataset. */
	private double m_SumInstances = 0;

	/** The number of attributes, including the class. 属性数量，包括类，如果只看属性需要-1 */
	private int m_NumAttributes;

	/** The number of class values. 类的数量 */
	protected int m_NumClasses;

	/** The index of the class attribute.类索引 */
	private int m_ClassIndex;

	/** The counts for each class value.每一个类值的数量 */
	private double[] m_Priors;

	/**
	 * The sums of attribute-class counts. m_CondiPriors[c][k] is the same as
	 * m_CondiCounts[c][k][k] 统计[类][属性]的数量，例如count(sunny,yes)
	 */
	private long[][] m_CondiPriors;

	/**
	 * For m_NumClasses * m_TotalAttValues * m_TotalAttValues.
	 * 统计[类][属性][属性]的数量，例如count(sunny,hot,yes)
	 */
	private long[][][] m_CondiCounts;

	/**
	 * The number of values for all attributes, not including class. 不包括类的所有的属性数量
	 */
	private int m_TotalAttValues;

	/**
	 * The starting index (in m_CondiCounts matrix) of each attribute.
	 * 属性的开始索引表，如outlook是第一个属性。索引为0，0368
	 */
	private int[] m_StartAttIndex;

	/** The number of values for each attribute. 每一个属性的数量，3322 */
	private int[] m_NumAttValues;

	/**
	 * The counts (frequency) of each attribute value for the dataset. Here for
	 * security, but it can be used for weighting.
	 */
	/** Count for P(ai, aj).两个属性同时发生的数量 */
	private int[][] AandB;
	private int[] A;// 每个属性值有几个

	/** The Smoothing parameter for M-estimation */
	private final double SMOOTHING = 5.0;

	/** The matrix of conditional mutual information 互信息矩阵 */
	private double[][] m_CondiMutualInfo;

	/** The minimum item in the matrix of conditional mutual information */
	private double EPSILON = 1.0E-4;

	/**
	 * The array to keep track of which attribute has which parent. (Tree) 最大生成树父节点
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
	private int m_Estimation = 0;

	/** If set, delete all the instances with any missing value (default: false). */
	private boolean m_DelMissing = false;

	/**
	 * Choose the root node for building the maximum spanning tree (default: m_Root
	 * = -1, i.e., set by random). 最大生成树根节点
	 */
	private int m_Root = -1;

	/**
	 * The number of arcs in current traning dataset, only for toString()
	 * 当前训练数据集中的弧数，仅用于toString()
	 */
	private int m_Arcs = 0;

	/**
	 * The number of instances with missing values, only for toString().
	 * 缺少值的实例数，仅用于toString()
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
	}
	// End of globalInfo()

	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if the classifier has not been generated successfully
	 */

	public void prim() {
		int n = m_CondiMutualInfo.length;
		int[] ss = new int[n];
		double[] highcost = new double[n];
		m_Parents[0] = -1;// 0为根节点，没有父节点
		highcost[0] = -1;// 只有0没有权重
		// 便利0点到所有点
		for (int i = 1; i < n; i++) {
			ss[i] = 0;
			m_Parents[i] = -1;// 除了0意外均没有父节点
			// 0到各个点权值
			highcost[i] = m_CondiMutualInfo[0][i];
		}
		int maxin = 0;
		for (int i = 0; i < n; i++) {
			double max = -1;
			for (int j = 0; j < n; j++) {
				if (highcost[j] != -1 && highcost[j] > max) {
					max = highcost[j];
					maxin = j;
				}
			}
			m_Parents[maxin] = ss[maxin];

			// 将已加入MST的对应的权值赋值为-1
			highcost[maxin] = -1;
			for (int k = 0; k < n; k++) {
				// 用maxin顶点到各个顶点的权值比较highCosts数组的值，若较大则替换，并更新起点为maxin
				double newCost = m_CondiMutualInfo[maxin][k];
				if (highcost[k] != -1 && newCost > highcost[k]) {
					highcost[k] = newCost;
					// 更新k的起点为maxIndex
					ss[k] = maxin;
				}
			}
		}

	}

	public void buildClassifier(Instances instances) throws Exception {

		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

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
		if (m_DelMissing) {
			for (int i = 0; i < m_NumAttributes; i++) {
				Attribute attribute = (Attribute) instances.attribute(i);
				m_Instances.deleteWithMissing(attribute);
			}
			// 有缺失值的实例熟练
			m_NumOfMissings = m_NumInstances - m_Instances.numInstances();
		}

		// Initialize the following counters有效类值样本数
		m_SumInstances = 0;

		// Copy the instances

		m_Instances = new Instances(instances);
		m_NumInstances = m_Instances.numInstances();

		// Initialize the following counters
		m_SumInstances = 0;
		m_TotalAttValues = 0;
		m_StartAttIndex = new int[m_NumAttributes];
		m_NumAttValues = new int[m_NumAttributes - 1];
		int t = 0;
		for (int i = 0; i < m_NumAttributes - 1; i++) {
			m_StartAttIndex[i] = t;
			m_NumAttValues[i] = m_Instances.attribute(i).numValues();
			t += m_NumAttValues[i];
			m_TotalAttValues += m_NumAttValues[i];

		}
		A = new int[m_TotalAttValues];

		// System.out.printf("%s\t",m_Instances.attribute(0).value(0));
		// System.out.printf("%s\t",m_Instances.attribute(0).index());

		// m_Priors矩阵正确
		m_Priors = new double[m_NumClasses];// 类的数量计数
		// 计算m_Priors，计数过程
		for (int i = 0; i < m_NumInstances; i++) {
			Instance in1 = m_Instances.instance(i);
			m_SumInstances++;
			// System.out.printf("%d\t",(int)in1.classValue());
			m_Priors[(int) in1.classValue()]++;
		}
		// for(int i=0;i<m_NumClasses;i++)
		// System.out.printf("%f\n",m_Priors[i]);
		//
		// 计算m_CondiPriors，计数过程
		m_CondiPriors = new long[m_NumClasses][m_TotalAttValues];
		int ind1 = 0;
		for (int i = 0; i < m_NumInstances; i++) {
			Instance in2 = m_Instances.instance(i);
			for (int j = 0; j < m_NumAttributes - 1; j++) {
				Attribute at = in2.attribute(j);
				ind1 = m_StartAttIndex[j] + (int) in2.value(at);
				if (!in2.isMissing(at))
					instances.deleteWithMissing(j);
				A[ind1]++;
				m_CondiPriors[(int) in2.classValue()][ind1]++;

			}
		}
		// 测试A矩阵，正确
		// for(int i=0;i<m_TotalAttValues;i++)
		// System.out.printf("%d\n",A[i]);

		// 测试m_CondiPriors矩阵，正确
		// for(int i=0;i<m_NumClasses;i++)
		// System.out.printf("%d\t",m_CondiPriors[i][0]);

		// 计算m_CondiCounts，计数过程
		m_CondiCounts = new long[m_NumClasses][m_TotalAttValues][m_TotalAttValues];
		int ind2 = 0, ind3 = 0;
		for (int i = 0; i < m_NumInstances; i++) {
			for (int j = 0; j < m_NumAttributes - 2; j++) {
				for (int k = j + 1; k < m_NumAttributes - 1; k++) {
					Instance in3 = m_Instances.instance(i);
					Attribute at1 = in3.attribute(j);
					Attribute at2 = in3.attribute(k);
					ind2 = m_StartAttIndex[j] + (int) in3.value(at1);
					ind3 = m_StartAttIndex[k] + (int) in3.value(at2);
					m_CondiCounts[(int) in3.classValue()][ind2][ind3]++;
					m_CondiCounts[(int) in3.classValue()][ind3][ind2]++;
				}
			}
		}
		// 测试m_CondiCounts矩阵，正确
		// for(int i=0;i<m_NumClasses;i++)
		// System.out.printf("%d\t",m_CondiCounts[i][0][3]);

		// Allocate space for conditional mutual information, including class.
		// 建立计算互信息矩阵
		// int lop=0;
		int atin1 = 0, atin2 = 0;
		m_CondiMutualInfo = new double[m_NumAttributes][m_NumAttributes];
		for (int a1 = 0; a1 < m_NumAttributes - 1; a1++) {// 第a1个属性,相当于公式里面的ai
			for (int a2 = a1 + 1; a2 < m_NumAttributes - 1; a2++) {// 第a2个属性,相当于公式里面的aj
				if (a1 == a2) {
					m_CondiMutualInfo[a1][a2] = 0.0;
					continue;
				}
				double sum = 0;
				for (int cc = 0; cc < m_NumClasses; cc++) {
					for (int ind4 = 0; ind4 < m_NumAttValues[a1]; ind4++) {// 第1个属性具体是哪个值，比如outlook里面的sunny，overcast，rainy都要便利到
						for (int ind5 = 0; ind5 < m_NumAttValues[a2]; ind5++) {// 第2个属性具体是哪个值，比如里面的hot,mild,cold都要便利到

							atin1 = m_StartAttIndex[a1] + ind4;
							atin2 = m_StartAttIndex[a2] + ind5;
							double countc = m_Priors[cc];
							double countai_c = m_CondiPriors[cc][atin1];
							double countaj_c = m_CondiPriors[cc][atin2];
							double countaijc = m_CondiCounts[cc][atin1][atin2];
							double valueai = m_NumAttValues[a1];
							double valueaj = m_NumAttValues[a2];
							double valuec = m_NumClasses;

							double pij_c = (countaijc + 1) / (countc + valueai * valueaj);
							double pi_c = (countai_c + 1) / (countc + valueai);
							double pj_c = (countaj_c + 1) / (countc + valueaj);
							// double pc=(countc+1)/(m_NumInstances+valuec);
							// double pijc=pij_c*pc;
							double pijc = (countaijc + 1) / (m_NumInstances * 1.0 + valueai * valueaj * valuec);
							// if(lop==0){
							// System.out.printf("%f\n",pij_c);
							// System.out.printf("%f\n",pi_c);
							// System.out.printf("%f\n",pj_c);
							// System.out.printf("%f\n",pijc);
							// lop++;
							// }
							sum += pijc * Math.log(pij_c / (pi_c * pj_c));

						} // cc

					} // a2in

				} // a1in
				// 对角线的元素不考虑了，应该置0
				m_CondiMutualInfo[a1][a2] = sum;
				m_CondiMutualInfo[a2][a1] = sum;

			} // a2
		} // a1
		// //验证互信息矩阵，正确
		// for(int i=0;i<instances.numAttributes();i++){
		// for(int j=0;j<instances.numAttributes();j++){
		// System.out.printf("%f\t", m_CondiMutualInfo[i][j]);//输出互信息矩阵
		// }
		// System.out.printf("\n");
		// }
		// 建立最大生成树
		m_Parents = new int[m_CondiMutualInfo.length];
		prim();
		// 生成树测试
		// for(int i=0;i<m_CondiMutualInfo.length;i++){
		// System.out.printf(i+"——"+m_Parents[i]+"\n");
		// }

	}// build classifier
	// 全部采用Laplace估计

	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] probs = new double[instance.numClasses()];// 每一类发生的概率

		for (int j = 0; j < m_NumClasses; j++) {// 计算属于第J类概率
			probs[j] = 1.0;// 初始概率为1
			double pc = (m_Priors[j] + 1) / (double) (m_SumInstances + m_NumClasses);

			for (int i = 0; i < m_NumAttributes - 1; i++) {// 第j类条件下遍历属性
				Attribute ai = instance.attribute(i);// 实例的第i个属性
				if (!instance.isMissing(ai)) {// 第ai存在
					int aia = (int) (m_StartAttIndex[i] + instance.value(i));

					double valuaB = (double) m_NumAttValues[i];
					double countY = (double) m_Priors[j];
					double countBY = (double) m_CondiPriors[j][aia];
					double pBY = (double) (countBY + 1) / (countY + valuaB);

					if (m_Parents[i] != -1) {// 第i个属性有父节点
						int aj = m_Parents[i];// 找到其父节点的属性
						int aja = (int) (m_StartAttIndex[aj] + instance.value(aj));// 确定父节点属性的索引

						double countAY = (double) m_CondiPriors[j][aja];
						double countABY = (double) m_CondiCounts[j][aja][aia];
						double pBAY = (countABY + 1) / (countAY + valuaB);
						// probs[j]*=(double)((m_CondiCounts[j][aja][aia]+1)/(m_CondiPriors[j][aja]+m_NumAttValues[i]));
						probs[j] *= pBAY;
					} else {
						// probs[j]*=(double)((m_CondiPriors[j][aia]+1)/(m_Priors[j]+m_NumAttValues[i]));
						probs[j] *= pBY;
					}
				} // miss
			} // att
			probs[j] *= pc;
			System.out.printf("概率" + "%f\n", probs[j]);
			// probs[j]=pt;
		}
		Utils.normalize(probs);
		return probs;
	}

	// 用alpha

	// double alpha = 0.0, t = 0.0;
	//
	// public double[] distributionForInstance(Instance instance) throws Exception {
	// double[] probs = new double[instance.numClasses()];// 每一类发生的概率
	// for (int j = 0; j < m_NumClasses; j++) {// 计算属于第J类概率
	// probs[j] = 1;// 初始概率为1
	// double pc = (m_Priors[j] + 1) / (double) (m_NumInstances + m_NumClasses);
	// double pt = 1;// 初始概率为1
	// for (int i = 0; i < m_NumAttributes - 1; i++) {// 第j类条件下遍历属性
	// Attribute ai = instance.attribute(i);// 实例的第i个属性
	// if (!instance.isMissing(ai)) {// 第ai存在
	// if (ai.isNominal()) {
	// int aia = m_StartAttIndex[i] + (int) (instance.value(ai));//
	// 具体到ai这个属性的取哪个值，比如sunny是0
	// double countB = A[aia];
	// // System.out.printf("第"+aia+"属性有"+A[aia]+"\n");
	// double pB = countB / m_NumInstances;
	// if (m_Parents[i] != -1) {// 第i个属性有父节点，ct(B|AY)
	// int aj = m_Parents[i];// 找到其父节点的属性，找到A所属于的属性
	// int aja = m_StartAttIndex[aj] + (int) instance.value(aj);// 确定父属性的具体的某个属性取值
	// double countAY = (double) m_CondiPriors[j][aja];
	// double countABY = (double) m_CondiCounts[j][aja][aia];
	// double pBAY = countABY / countAY;
	// t = m_NumInstances * (countAY / m_NumInstances);
	// alpha = countAY / (countAY + SMOOTHING);
	// // System.out.printf("%f\n",alpha);
	// pt *= alpha * pBAY + (1 - alpha) * pB;
	// } else {
	// double countY = (double) m_Priors[j];
	// alpha = countY / (countY + SMOOTHING);
	// // System.out.printf("%f\n",alpha);
	// double countBY = (double) m_CondiPriors[j][aia];
	// double pBY = countBY / countY;
	// pt *= alpha * pBY + (1 - alpha) * pB;
	// }
	// }
	// } // miss
	// } // att
	// probs[j] = pc * pt;
	// // System.out.printf("概率"+"%f\n",probs[j]);
	// }
	// // Normalize probabilities
	// // Utils.normalize(probs);
	// return probs;
	// }

	public static void main(String args[]) {
		runClassifier(new tan_z(), args);
	}

}