# exp1.go lcm & gcd
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
func main() {
var n int
fmt.Print("Enter the number of levels: ")
fmt.Scan(&n)
for i:=1; i<=n; i++ {
for j:=n; j<=n-i; j++ {
fmt.Print(" ")
}
for j:=1;j<=i;j++ {
fmt.Print(j)
}
for j:=i-1; j>=1; j-- {
fmt.Print(j)
}
fmt.Println()
}
}
```
## output
```
Enter the number of levels: 5
1
121
12321
1234321
123454321
```
# week 5 print Floyd's Triangle
```go
package main
import "fmt"
func main() {
var rows int
var temp int=1
fmt.Print("Enter number of rows:")
fmt.Scan(&rows)
for i:=1;i<=rows;i++ {
for k:=1;k<=i;k++ {
fmt.Printf("%d",temp)
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
