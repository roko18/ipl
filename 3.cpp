#include <iostream>
#include <omp.h>
#include <ctime>
#include <cstdlib>
#include <climits>  // For INT_MAX, INT_MIN

using namespace std;

void min(int *arr, int n)
{
   int min_val = INT_MAX;  // Use INT_MAX to ensure proper reduction for min
   int i;

   cout << endl;
   #pragma omp parallel for reduction(min : min_val)
   for (i = 0; i < n; i++)
   {
      // Debugging: showing thread id and iteration index
      // cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
      if (arr[i] < min_val)
      {
         min_val = arr[i];
      }
   }
   cout << "\n\nmin_val = " << min_val << endl;
}

void max(int *arr, int n)
{
   int max_val = INT_MIN;  // Use INT_MIN to ensure proper reduction for max
   int i;

   #pragma omp parallel for reduction(max : max_val)
   for (i = 0; i < n; i++)
   {
      // Debugging: showing thread id and iteration index
      // cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
      if (arr[i] > max_val)
      {
         max_val = arr[i];
      }
   }
   cout << "\n\nmax_val = " << max_val << endl;
}

void avg(int *arr, int n)
{
   int i;
   float avg = 0.0, sum = 0;

   #pragma omp parallel for reduction(+:sum)
   for (i = 0; i < n; i++)
   {
      sum += arr[i];
      // Debugging: showing thread id and iteration index
      // cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
   }

   cout << "\n\nSum = " << sum << endl;
   avg = sum / n;
   cout << "\nAverage = " << avg << endl;
}

int main()
{
   omp_set_num_threads(4);  // Set the number of threads
   int n, i;

   cout << "Enter the number of elements in the array: ";
   cin >> n;

   int arr[n];  // Create an array of size n

   srand(time(0));  // Initialize random seed
   for (i = 0; i < n; ++i)
   {
      arr[i] = rand() % 100;  // Random values between 0 and 99
   }

   cout << "\nArray elements are: ";
   for (i = 0; i < n; i++)
   {
      cout << arr[i] << " ";
   }
   cout << endl;

   min(arr, n);
   max(arr, n);
   avg(arr, n);

   return 0;
}

