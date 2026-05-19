// Quick sort implementation (in-place)
function quickSort(arr, left = 0, right = arr.length - 1) {
  if (left >= right) {
    return;
  }
  const pivotIndex = partition(arr, left, right);
  quickSort(arr, left, pivotIndex - 1);
  quickSort(arr, pivotIndex + 1, right);
}

function partition(arr, left, right) {
  const pivotValue = arr[right];
  let i = left - 1;
  for (let j = left; j < right; j++) {
    if (arr[j] <= pivotValue) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]]; // swap
    }
  }
  // Place pivot in correct position
  i++;
  [arr[i], arr[right]] = [arr[right], arr[i]];
  return i;
}

// Example usage:
// const arr = [4, 6, 2, 5, 3];
// quickSort(arr);
// console.log(arr); // [2, 3, 4, 5, 6]
