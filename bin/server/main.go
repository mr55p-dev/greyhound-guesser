package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"text/template"

	"github.com/labstack/echo"
)

type FormControls struct {
	TrapNumbers []int
}

func ReadTemplate(data any, files ...string) string {
	out := new(bytes.Buffer)
	t := template.Must(
		template.ParseFiles(files...),
	)
	t.Execute(out, data)
	return out.String()
}

func main() {
	e := echo.New()
	e.Logger.Info("Starting server")
	e.Static("/", "public")
	e.GET("/", func(c echo.Context) error {
		c.Logger().Debug("Index request received")
		html := ReadTemplate(
			FormControls{TrapNumbers: []int{0, 1, 2, 3, 4, 5}},
			"templates/index.html",
			"templates/common.html",
		)
		return c.HTML(http.StatusOK, html)
	})

	e.POST("/submit", func(c echo.Context) error {
		c.Logger().Debug("Submit request received")
		reqBuf := new(bytes.Buffer)
		io.Copy(reqBuf, c.Request().Body)
		reqBody := reqBuf.Bytes()
		c.Logger().Debugf("Request body: %s", string(reqBody))
		ctype := c.Request().Header.Get("Content-Type")
		c.Logger().Debugf("Content type submitted as %s", ctype)

		res, err := http.Post("http://127.0.0.1:5000/predict", ctype, bytes.NewReader(reqBody))
		if err != nil {
			return c.String(http.StatusInternalServerError, fmt.Sprintf("Failed to get inference: %s", err.Error()))
		}
		c.Logger().Info("Received response from inference server")
		defer res.Body.Close()
		resData := new(bytes.Buffer)
		io.Copy(resData, res.Body)
		c.Logger().Debug("Copied response body")

		if res.StatusCode != 200 {
			c.Logger().Warnf("Got bad response %s from inference server", res.Status)
			return c.String(http.StatusInternalServerError, fmt.Sprintf("Error fetching inference: (%s) %s", res.Status, resData.String()))
		}
		out := make([]float32, 5)
		c.Logger().Debug("Trying to unmarshal response")
		json.Unmarshal(resData.Bytes(), &out)
		c.Logger().Debugf("Received response from model server: %v", out)

		html := ReadTemplate(
			out,
			"templates/results.html",
			"templates/common.html",
		)
		return c.HTML(http.StatusOK, html)
	})

	e.Logger.SetLevel(0)
	log.Fatal(e.Start(":3000"))
}

