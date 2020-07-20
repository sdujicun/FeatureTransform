package weka.filters.timeseries.shapelet_transforms.sshvg.hvg;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fileIO.DataSets;
import weka.core.Instance;
import weka.core.Instances;

public class HVG{
//	public int[][] convertHVG1(Instance series) {
//		double[] data = series.toDoubleArray();
//		int c = series.classIndex();
//		if (c >= 0) {
//			double[] temp;
//			temp = new double[data.length - 1];
//			System.arraycopy(data, 0, temp, 0, c); // assumes class attribute is in last index											 
//			data = temp;
//		}
//
//		int length = data.length;
//		int[][] visibility = new int[length][length];
//		for (int i = 0; i < length - 1; i++) {
//			double xi = data[i];
//			for (int j = i + 1; j < length; j++) {
//				double xj = data[j];
//				double min = Math.min(xi, xj);
//				double temp = (xj - xi) / (j - i);
//				boolean visible = true;
//				for (int k = i + 1; k < j; k++) {
//
//					double x = xi + (k - i) * temp;
//					if (x <= data[k] || data[k] >= min) {
//						visible = false;
//						break;
//					}
//				}
//				if (visible) {
//					visibility[i][j] = 1;
//					visibility[j][i] = 1;
//				}
//
//			}
//		}
//		return visibility;
//	}

	public int[][] convertHVG(Instance series) {

		double[] data = series.toDoubleArray();
		int c = series.classIndex();
		if (c >= 0) {
			double[] temp;
			temp = new double[data.length - 1];
			System.arraycopy(data, 0, temp, 0, c); // assumes class attribute is in last index
			data = temp;
		}
		int length = data.length;
		int[][] visibility = new int[length][length];
		
		int leftIndex = 0;
		int rightIndex = data.length - 1;

		getMaxPointVisibility(data, visibility, leftIndex, rightIndex);

		return visibility;
	}

	private void getMaxPointVisibility(double[] data, int[][] visibility,
			int leftIndex, int rightIndex) {
		int maxIndex = findMaxIndexRight(data, leftIndex, rightIndex);

		// 处理左边
		int leftLength = maxIndex - leftIndex;

		if (1 == leftLength) {
			visibility[maxIndex][leftIndex] = 1;
			visibility[leftIndex][maxIndex] = 1;
			
			
			
		} else if (1 < leftLength) {
			processLeft(maxIndex, data, visibility, leftIndex, maxIndex - 1);
		}
		// 处理右边
		int rightLength = rightIndex - maxIndex;
		if (1 == rightLength) {
			visibility[maxIndex][rightIndex] = 1;
			visibility[rightIndex][maxIndex] = 1;
			
		} else if (1 < rightLength) {
			processRight(maxIndex, data, visibility, maxIndex + 1, rightIndex);
		}		
		if (1 < leftLength) {
			getMaxPointVisibility(data, visibility, leftIndex, maxIndex - 1);
		}
		if (1 < rightLength) {
			getMaxPointVisibility(data, visibility, maxIndex + 1, rightIndex);
		}

	}

	// 查找最大值索引，如果最大值有多个，则返回最左边的一个
	private int findMaxIndexLeft(double[] data, int leftIndex, int rightIndex) {
		int maxIndex = leftIndex;
		for (int i = leftIndex + 1; i <= rightIndex; i++) {
			if (data[i] > data[maxIndex]) {
				maxIndex = i;
			}
		}

		return maxIndex;
	}

	// 查找最大值索引，如果最大值有多个，则返回最右边的一个
	private int findMaxIndexRight(double[] data, int leftIndex, int rightIndex) {
		int maxIndex = leftIndex;
		for (int i = leftIndex + 1; i <= rightIndex; i++) {
			if (data[i] >= data[maxIndex]) {
				maxIndex = i;
			}
		}

		return maxIndex;
	}

	// 处理最高点左边，只看右边的新的最高点
	private void processLeft(int maxIndex, double[] data, int[][] visibility,
			int leftIndex, int rightIndex) {
		int subMaxIndex = findMaxIndexRight(data, leftIndex, rightIndex);
		visibility[subMaxIndex][maxIndex] = 1;
		visibility[maxIndex][subMaxIndex] = 1;
		
		
		int rightLength = rightIndex - subMaxIndex;
		if (1 == rightLength) {
			visibility[rightIndex][maxIndex] = 1;
			visibility[maxIndex][rightIndex] = 1;
		} else if (1 < rightLength) {
			processLeft(maxIndex, data, visibility, subMaxIndex + 1, rightIndex);
		}

	}

	// 处理最高点右边，只看左边的新的最高点
	private void processRight(int maxIndex, double[] data, int[][] visibility,
			int leftIndex, int rightIndex) {
		int subMaxIndex = findMaxIndexLeft(data, leftIndex, rightIndex);
		visibility[subMaxIndex][maxIndex] = 1;
		visibility[maxIndex][subMaxIndex] = 1;
		
		int leftLength = subMaxIndex - leftIndex;
		if (1 == leftLength) {
			visibility[leftIndex][maxIndex] = 1;
			visibility[maxIndex][leftIndex] = 1;
		} else if (1 < leftLength) {
			processRight(maxIndex, data, visibility, leftIndex, subMaxIndex - 1);
		}

	}


	public int[] statisticsDegree(Instance series) {
		// remove class attribute if needed
		int[][] visibility = convertHVG(series);

		int length = visibility.length;
		int[] number = new int[length];
		for (int i = 0; i < length - 1; i++) {
			for (int j = 0; j < length - 1; j++) {
				number[i] += visibility[i][j];
			}
		}
		return number;
	}

	public int[] calcuateAbsoluteDifference(Instance series) {
		int[] HVGNumber = statisticsDegree(series);
		int[] diff = new int[HVGNumber.length];
		diff[0] = 0;
		for (int i = 1; i < diff.length; i++) {
			diff[i] = HVGNumber[i] - HVGNumber[i - 1];
		}
		return diff;
	}

	public int[] getSectionPointIndex(Instance series, int minDiff) {
		List<Integer> list = new ArrayList<Integer>();
		list.add(0);
		int[] diff = calcuateAbsoluteDifference(series);
		for (int i = 1; i < diff.length - 1; i++) {
			if (diff[i] >= minDiff) {
				list.add(i);
			}
		}
		list.add(diff.length - 1);
		int[] index = new int[list.size()];
		for (int i = 0; i < index.length; i++) {
			index[i] = list.get(i);
		}
		return index;
	}


}
