# week 1 lcm & gcd
```go
package main

import "fmt"

func lcm(temp1 int, temp2 int) {
	var lcmnum int = 1
	if temp1 > temp2 {
		lcmnum = temp1
	} else {
		lcmnum = temp2
	}

	for {
		if lcmnum%temp1 == 0 && lcmnum%temp2 == 0 {
			fmt.Printf("LCM of %d and %d is %d", temp1, temp2, lcmnum)
			break
		}
		lcmnum++
	}
	return
}
func gcd(temp1 int, temp2 int) {
	var gcdnum int
	for i := 1; i <= temp1 && i <= temp2; i++ {
		if temp1%i == 0 && temp2%i == 0 {
			gcdnum = i
		}
	}
	fmt.Printf("GCD of %d and %d is %d", temp1, temp2, gcdnum)
	return
}

func main() {
	var n1, n2, action int
	fmt.Println("Enter two psitive integers:")
	fmt.Scanln(&n1)
	fmt.Scanln(&n2)
	fmt.Println("Enter 1 for LCM and 2 for GCD")
	fmt.Scanln(&action)
	switch action {
	case 1:
		lcm(n1, n2)
	case 2:
		gcd(n1, n2)
	}
}
```
## output
```
PS C:\Users\student\Desktop\66h7> go mod init exp1.go
go: creating new go.mod: module exp1.go
go: to add module requirements and sums:
        go mod tidy
PS C:\Users\student\Desktop\66h7> go run exp1.go
Enter two positive integers:
4
5
Enter 1 for LCM and 2 for GCD
1
LCM of 4 and 5 is 20
PS C:\Users\student\Desktop\66h7> go run exp1.go
Enter two positive integers:
5 50
Enter 1 for LCM and 2 for GCD
2
GCD of 5 and 0 is 0
PS C:\Users\student\Desktop\66h7> go run exp1.go
Enter two positive integers:
3
8
Enter 1 for LCM and 2 for GCD
2
GCD of 3 and 8 is 1
```
# week 2 print pyramid of numbers
```go
package main
import "fmt"
func main(){
              var n int
              fmt.Print("Enter the number of levels:")
              fmt.Scan(&n)
              for i:=1;i<=n;i++{
                  //print leading spaces
                  for j:=1;j<=n-i;j++{
                       fmt.Print(" ")
                  }
              //print numbers in increasing order
              for j:=1;j<=i;j++{
                  fmt.Print(j)
              }
               //print numbers in decreasing order
              for j:=i-1;j>=1;j--{
                  fmt.Print(j)
              }
           //move to next line
                fmt.Println()
            }
}
```
## Output
```
Enter the number of levels:
    1
   121
  12321
 1234321
123454321
```
# week 4 cal standard deviation in math pkg
```go
package main

import (
	"fmt"
	"math"
)
func main() {
	var num [10]float64
	var sum, mean, sd float64
	fmt.Println("**********enter 10 elements************")
	for i := 1; i <= 10; i++ {
		fmt.Printf("enter %d element:", i)
		fmt.Scan(&num[i-1])
		sum += num[i-1]
	}
	mean = sum / 10
	for j := 0; j < 10; j++ {
		sd += math.Pow(num[j]-mean, 2)
	}
	sd = math.Sqrt(sd / 10)
	fmt.Println("the sd is:", sd)
}
```
## output
```
**********enter 10 elements************
enter 1 element:2
enter 2 element:3
enter 3 element:4
enter 4 element:5
enter 5 element:6
enter 6 element:7
enter 7 element:8
enter 8 element:9
enter 9 element:10
enter 10 element:11
the sd is: 2.8722813232690143
```
# week 5 print Floyd's Triangle
```go
package main
import "fmt"
func main(){
              var rows int
              var temp int=1
              fmt.Print("Enter the number of rows:")
              fmt.Scan(&rows)
              for i:=1;i<=rows;i++{
                  for k:=1;k<=i;k++{
                       fmt.Print(" ",temp)
              temp++
                  }
fmt.Println(" ")
}
} 
```
## output
```
Enter number of rows: 4
1
2 3
4 5 6
7 8 9 10
```
# week 6 take user input and addition of two strings
```go
package main
import "fmt"
func main(){
  fmt.Print("enter first string :")
  var first string
  fmt.Scanln(&first)
  fmt.Print("enter second string :")
  var second string
  fmt.Scanln(&second)
  fmt.Print(first+second)
}
```
## output
```
enter first string :hello
enter second string :world
helloworld
```
# week 7 string palindrome
```go
package main
import "fmt"
func main(){
var number, remainder, temp int
var reverse int=10
fmt.Print("Enter any positive integer:")
fmt.Scan(&number)
temp=number
for{
remainder=number%10
reverse=reverse*10 + remainder
number%10
if(number==0){
break
}
}
if(temp==reverse){
fmt.Printf("%d is a palindrome",temp)
}
else
{
fmt.Printf("%d is not a palindrome",)
}
}
------------------------------------------------
package main
import(
"fmt"
"strings"
)
func main(){
var originalString string = "madam"
var reverseString string = ""
var length=len(originalString)

for i:=length-1;i>=0;i-- {
reverseString=reverseString + string(originalString[i])
}
if strings.ToLower(originalString)==strings.ToLower(reverseString){
fmt.Println("The given string is palindrome");
}else{
fmt.Println("The given string is not a palindrome");
}
}
```
## output
```
Enter any positive integer:24
24 is not a palindrome
Enter any positive integer:121
121 is a palindrome
```
# week 8 build a contact form
```go
<!DOCTYPE html>
<html>
<head>
<title> Simple Form Example </title>
</head>
<body>
<h2> Submit Form </h2>
<form action="/" method="post">
<label for="name"> Name:</label><br>
<input type="text" id="name" name="name" required> <br><br>

<label for="email"> Email:</label><br>
<input type="email" id="email" name="email" required> <br><br>

<label for="email"> Message:</label><br>
<input type="Message" id="Message" name="Message" required> <br><br>

<input type="reset" value="reset">
<input type="submit" value="Submit">
</form>
</body>
</html>
```
```
package main
import (
	"fmt"
	"net/http"
)
func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.ServeFile(w, r, "form.html")
			return
		}
		r.ParseForm() // Parse form data
		name := r.Form.Get("name")
		email := r.Form.Get("email")
		fmt.Println("Name:", name)
		fmt.Println("Email:", email)

		fmt.Fprintf(w, "Received form submission\nName: %s\nEmail: %s\n", name, email)
	})
	fmt.Println("Server is running on localhost:8080")
	http.ListenAndServe(":8080", nil) // Corrected port specification
}
```
# week 9 calculate average using arrays
```go
package main
import "fmt"
func main(){
var num[100] int
var temp, sum, avg int
fmt.Print("enter no of elements:")
fmt.Scanln(&temp)
for i:=0;i<temp;i++ {
fmt.Print("enter the numbers:")
fmt.Scanln(&num[i])
sum+=num[i]
}
avg=sum/temp
fmt.Printf("avg of %d numbers (s) is %d" , temp, avg)
}
```
## output
```
enter no of elements:5
enter the numbers:3
enter the numbers:2
enter the numbers:1
enter the numbers:6
enter the numbers:4
avg of 5 numbers (s) is 3
```
# week 10 delete duplicate element in array
```go
package main
import "fmt"

func removeDuplicate(arr[8]int) []int {
map_var := map[int]bool{}
	result := []int{}
	for e := range arr {
		if map_var[arr[e]] != true {
		   map_var[arr[e]] = true
		   result = append(result,arr[e])
		}
	}
	return result
}
func main() {
   arr := [8]int{1,2,2,4,4,5,7,5}
   fmt.Println("The unsorted array entered is:", arr)
   result := removeDuplicate(arr)
   fmt.Println("Thr array obtained after removing the duplicate values is:", result)
}
```
## output
```
The unsorted array entered is: [1 2 2 4 4 5 7 5]
Thr array obtained after removing the duplicate values is: [1 2 4 5 7]
```
# week 11 reverse array sort for integers and strings
```go
package main
import ("fmt"
"sort"
)
func main(){
fmt.Println("Integer Reverse sort")
num:=[]int { 50,20,10,35,62}
sort.Sort(sort.Reverse(sort.IntSlice(num)))
fmt.Println(num)
fmt.Println("String Reverse Sort")
text:=[] string{"India", "Australia", "Japan", "Germany"}
sort.Sort(sort.Reverse(sort.StringSlice(text)))
fmt.Println(text)
}
```
## output
```
Integer Reverse sort
[62 50 35 20 10]
String Reverse Sort
[Japan India Germany Australia]
```
# week12 contains, contains any, count, equal fold string functions
```go
package main
import (
"fmt"
"strings"
)
func main(){
fmt.Println(strings.ContainsAny("Germany","G"))
fmt.Println(strings.ContainsAny("Germany","g"))
fmt.Println(strings.Contains("Germany","Ger"))
fmt.Println(strings.Contains("Germany","ger"))
fmt.Println(strings.Contains("Germany","er"))
fmt.Println(strings.Count("cheese","e"))
fmt.Println(strings.EqualFold("Cat","cAt"))
fmt.Println(strings.EqualFold("India","Indiana"))
}
```
## output
```
true
false
true
false
true
3
true
false
```
# week 14 create multiple go routines 
```
package main
import (
	"fmt"
	"runtime"
	"sync"
)
func main() {
	runtime.GOMAXPROCS(3)
	var processTest sync.WaitGroup
	processTest.Add(3)
	
	go func() {
		defer processTest.Done()
		for i := 0; i < 3; i++ {
			for j := 15; j <= 25; j++ {
				fmt.Printf(" %d", j)
				if j == 18{
					fmt.Println()
				}
			}
		}
	}()

	go func() {
		defer processTest.Done()
		for j := 0; j < 3; j++ {
			for char := 'A'; char < 'A'+10; char++ {
				fmt.Printf("%c ", char)
				if char == 'F' {
					fmt.Println()
				}
			}
		}
	}()

	go func() {
		defer processTest.Done()
		for i := 0; i < 3; i++ {
			for j := 0; j <= 15; j++ {
				fmt.Printf(" %d", j)
				if j == 10 {
					fmt.Println()
				}
			}
		}
	}()
	processTest.Wait()
}
```
## output
```
15 16 17 18
 19 20 21 22 23 24A B C D E F
G H I J A B C D E F
G H I J A B C D E F
G H I J  25 0 1 2 3 4 5 6 7 8 9 10
 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10
 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10
 11 12 13 14 15 15 16 17 18
 19 20 21 22 23 24 25 15 16 17 18
 19 20 21 22 23 24 25
```
