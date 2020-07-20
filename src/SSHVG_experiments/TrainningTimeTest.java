package SSHVG_experiments;

import java.io.File;
import java.util.ArrayList;

import tsc_algorithms.LearnShapelets;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.core.shapelet.Shapelet;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.fss.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.sshvg.ShapeletTransformWithHVG;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import fileIO.DataSets;

public class TrainningTimeTest {
	public static void main(String[] args) throws Exception {

		
		String[] problems= { 
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

		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i]+"\t");
			trainTimeForSSHVG(problems[i]);
			trainTimeForFSS(problems[i]);
			trainTimeForLS(problems[i]);
			trainTimeForST(problems[i]);			
			System.out.println();			
		}
		
	}

	public static void trainTimeForSSHVG(String problem) throws Exception {
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

		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
	}
	
	public static void trainTimeForFSS(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransformWithSubclassSampleAndLFDP transform = new ShapeletTransformWithSubclassSampleAndLFDP();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
	}
	public static void trainTimeForST(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		FullShapeletTransform transform = new FullShapeletTransform();
		transform.setRoundRobin(true);
		// construct shapelet classifiers.
		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
		long d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		//Instances tranTest = transform.process(test);
		ArrayList<Shapelet> sh = transform.getShapelets();
		long d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
	}
	public static void trainTimeForLS(String problem) throws Exception {

		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		     

        LearnShapelets ls = new LearnShapelets();
        ls.setSeed(0);
        ls.setParamSearch(true);
        
		long d1, d2;
		d1 = System.nanoTime();
		ls.buildClassifier(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
	}
	
}
