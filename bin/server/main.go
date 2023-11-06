package main

import (
	"bytes"
	"log"
	"net/http"
	"text/template"

	"github.com/labstack/echo"
)

type FormControls struct {
	TrapNumbers []int
}

func main() {
	log.Print("Hello, world")
	e := echo.New()
	e.Static("/", "public")
	e.GET("/", func(c echo.Context) error {
		e.Logger.Print("Request received")
		templ := template.Must(template.ParseFiles("templates/index.html", "templates/common.html"))
		out := new(bytes.Buffer)
		templ.Execute(out, FormControls{
			TrapNumbers: []int{1, 2, 3, 4, 5, 6},
		})
		return c.HTML(http.StatusOK, out.String())
	})

	e.POST("/submit", func(c echo.Context) error {
		e.Logger.Print("Submission received")
		templ := template.Must(template.ParseFiles("templates/results.html", "templates/common.html"))
		out := new(bytes.Buffer)
		templ.Execute(out, nil)
		return c.HTML(http.StatusOK, out.String())
	})

	log.Fatal(e.Start(":3000"))
}
