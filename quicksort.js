// 快速排序（原地排序）
function quickSort(arr, left = 0, right = arr.length - 1) {
  if (left >= right) {
    return; // 如果只剩一个元素，说明已经有序
  }
  const pivotIndex = partition(arr, left, right); // 分区，并获取基准最终位置
  quickSort(arr, left, pivotIndex - 1); // 递归排序左侧
  quickSort(arr, pivotIndex + 1, right); // 递归排序右侧
}

function partition(arr, left, right) {
  const pivotValue = arr[right]; // 选取最右元素作为基准
  let i = left - 1; // i 指向比基准小的最后一个元素的位置
  for (let j = left; j < right; j++) {
    if (arr[j] <= pivotValue) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]]; // 交换
    }
  }
  // 将基准放到正确的位置
  i++;
  [arr[i], arr[right]] = [arr[right], arr[i]];
  return i; // 返回基准的下标
}

// 示例用法：
// const arr = [4, 6, 2, 5, 3];
// quickSort(arr);
// console.log(arr); // [2, 3, 4, 5, 6]
