dberns@bluefire:~/Projects/github/DanielBerns/AI-course/alexnet$ uv run python serve.py --instance_path ~/Info/alexnet/alpha --host 0.0.0.0 --port 8000
[*] Starting AlexNet Web Studio Server...
[*] Instance path   : /home/dberns/Info/alexnet/alpha
[*] Config          : /home/dberns/Info/alexnet/alpha/config.yaml
[*] Network Endpoint: http://0.0.0.0:8000
INFO:     Started server process [25043]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:44146 - "GET / HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/uvicorn/protocols/http/httptools_impl.py", line 422, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/uvicorn/middleware/proxy_headers.py", line 63, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/fastapi/applications.py", line 1163, in __call__
    await super().__call__(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/applications.py", line 96, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py", line 186, in __call__
    raise exc
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/routing.py", line 670, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/fastapi/routing.py", line 2734, in app
    await route.handle(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/fastapi/routing.py", line 1281, in handle
    await super().handle(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/routing.py", line 280, in handle
    await self.app(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/fastapi/routing.py", line 158, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/fastapi/routing.py", line 144, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/fastapi/routing.py", line 706, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/fastapi/routing.py", line 352, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/src/web/app.py", line 44, in index
    return templates.TemplateResponse("inference.html", {"request": request, "active_page": "inference"})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/templating.py", line 148, in TemplateResponse
    template = self.get_template(name)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/starlette/templating.py", line 115, in get_template
    return self.env.get_template(name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/jinja2/environment.py", line 1016, in get_template
    return self._load_template(name, globals)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/jinja2/environment.py", line 964, in _load_template
    template = self.cache.get(cache_key)
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/jinja2/utils.py", line 477, in get
    return self[key]
           ~~~~^^^^^
  File "/home/dberns/Projects/github/DanielBerns/AI-course/alexnet/.venv/lib/python3.12/site-packages/jinja2/utils.py", line 515, in __getitem__
    rv = self._mapping[key]
         ~~~~~~~~~~~~~^^^^^
TypeError: unhashable type: 'dict'

