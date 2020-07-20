package SSHVG_experiments;

import java.io.File;

import utilities.ClassifierTools;


import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.sshvg.ShapeletTransformWithHVG;
import weka.filters.timeseries.shapelet_transforms.sshvg.ShapeletTransformWithHVGNoSample;
import weka.filters.timeseries.shapelet_transforms.sshvg.ShapeletTransformWithSample;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import fileIO.DataSets;

public class SpeedupStrategyTest {
	public static void main(String[] args) throws Exception {

		String[] problems = { 
				"Adiac", // 390,391,176,37
				"Beef", // 30,30,470,5
				"ChlorineConcentration", 
				"Coffee", // 28,28,286,2
				"DiatomSizeReduction", // 16,306,345,4
				"ItalyPowerDemand", // 67,1029,24,2
				"Lightning7", // 70,73,319,7
				"MedicalImages", // 381,760,99,10
				"MoteStrain", // 20,1252,84,2
				"Symbols", // 25,995,398,6
				"Trace", // 100,100,275,4
				"TwoLeadECG", // 23,1139,82,2
		};		
		System.out.println(problems.length);
		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i] + "\t");
			testSSHVG(problems[i]);
			testSSHVG_S(problems[i]);
			testSSHVG_P(problems[i]);
			testST(problems[i]);
			System.out.println();
			
		}
	}

	public static void testSSHVG(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;
		
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransformWithHVG transform = new ShapeletTransformWithHVG();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		//Shapelet Selection Time
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
		
		Instances tranTest = transform.process(test);
		
		
		//Accuracy
		WeightedEnsemble we = new WeightedEnsemble();
		we.buildClassifier(tranTrain);
		double accuracy = ClassifierTools.accuracy(tranTest, we);
		System.out.print(accuracy + "\t");


	}
	
	public static void testSSHVG_S(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		
		
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransformWithSample transform = new ShapeletTransformWithSample();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		//Shapelet Selection Time
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
		
		Instances tranTest = transform.process(test);
		
		
		//Accuracy
		WeightedEnsemble we = new WeightedEnsemble();
		we.buildClassifier(tranTrain);
		double accuracy = ClassifierTools.accuracy(tranTest, we);
		System.out.print(accuracy + "\t");
	}
	
	public static void testSSHVG_P(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		
		
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransformWithHVGNoSample transform = new ShapeletTransformWithHVGNoSample();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		//Shapelet Selection Time
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
		
		Instances tranTest = transform.process(test);
		
		
		//Accuracy
		WeightedEnsemble we = new WeightedEnsemble();
		we.buildClassifier(tranTrain);
		double accuracy = ClassifierTools.accuracy(tranTest, we);
		System.out.print(accuracy + "\t");


	}
	
	
	public static void testST(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransform transform = new ShapeletTransform();
		transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		//Shapelet Selection Time
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
				
		
		//The Accuracy of ST are take from Hills, J., Lines, J., Baranauskas, E., Mapp, J., Bagnall, A.: Classication of time series by shapelet transformation. Data Mining and Knowledge Discovery, 28(4),851-881 (2014)

	}
}
