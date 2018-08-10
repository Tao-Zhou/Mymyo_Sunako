#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int Maxsub(vector<int> Data) {
  int N = Data.size();
  int lomax = Data[0];
  int res = 0;
  for (int i = 1; i < N; i++) {
    if (lomax < 0) {
      lomax = Data[i];
      continue;
    }
    lomax += Data[i];
    res = max(lomax, res);
  }
  return res;
}
int help_cross(vector<int> data, int low, int mid, int high) {
  int left_index, right_index;
  int left_sum = INT16_MIN, right_sum = INT16_MIN;
  int sum = 0;
  for (int i = mid; i > low; i--) {
    sum += data[i];
    if (sum > left_sum) {
      left_sum = sum;
      left_index = i;
    }
  }
  sum = 0;
  for (int i = mid + 1; i < high; i++) {
    sum += data[i];
    if (sum > right_sum) {
      right_sum = sum;
      right_index = i;
    }
  }
  return max(max(left_sum, right_sum), left_sum + right_sum);
}
int help(vector<int> data, int low, int high) {
  if (low == high) {
    return data[low];
  } else {
    int mid = (low + high) / 2;
    int left_sum = help(data, low, mid);
    int right_sum = help(data, mid + 1, high);
    int cross_sum = help_cross(data, low, mid, high);
    if (left_sum >= right_sum && left_sum >= cross_sum) {
      return left_sum;
    } else if (right_sum >= left_sum && right_sum >= cross_sum) {
      return right_sum;
    } else {
      return cross_sum;
    }
  }
}
int Maxsub_re(vector<int> Data) {
  int res = help(Data, 0, Data.size() - 1);
  return res;
}
int main() {
  vector<int> data = {13, -3, -25, 20, -3,  -16, -23, 18,
                      20, -7, 12,  -5, -22, 15,  -4,  7};
  cout << Maxsub(data) << endl;
  cout << Maxsub_re(data) << endl;
}
